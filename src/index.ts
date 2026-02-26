/**
 * Cloudflare Worker: TFLite NER output → Wikidata entity resolution
 *
 * POST /resolve  — resolve a batch of NER entities to Wikidata QIDs
 * GET  /health   — liveness check
 * GET  /         — API documentation
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A single entity as emitted by a TFLite NER model */
export interface NEREntity {
  /** Surface form of the entity (e.g. "Barack Obama") */
  text: string;
  /** IOB/BIO label: PER | LOC | ORG | MISC (or any custom label) */
  label: string;
  /** Character offset – start (optional) */
  start?: number;
  /** Character offset – end (optional) */
  end?: number;
  /** Model confidence score 0-1 (optional) */
  score?: number;
}

/** The full JSON body that the worker accepts */
export interface NERInput {
  /** Array of entities from TFLite NER */
  entities: NEREntity[];
  /** Original document text (optional, echoed back) */
  text?: string;
  /** BCP-47 language code for Wikidata labels (default: "en") */
  language?: string;
  /** Max Wikidata candidates per entity (1-10, default: 3) */
  limit?: number;
}

/** One Wikidata candidate returned for an entity */
export interface WikidataCandidate {
  /** Wikidata QID (e.g. "Q76") */
  id: string;
  /** Human-readable label in the requested language */
  label: string;
  /** Short description (e.g. "44th president of the United States") */
  description?: string;
  /** Wikidata entity URL */
  url: string;
  /** Aliases in the requested language */
  aliases?: string[];
  /** Wikidata entity type hint derived from NER label */
  typeHint?: string;
}

/** One resolved entity: original NER fields + Wikidata candidates */
export interface ResolvedEntity extends NEREntity {
  /** Ordered list of Wikidata candidates (best match first) */
  wikidata: WikidataCandidate[];
  /** True when this result was served from the CF cache */
  cached?: boolean;
}

/** Full API response */
export interface ResolveResponse {
  ok: boolean;
  language: string;
  resolvedCount: number;
  entities: ResolvedEntity[];
  /** Original text echoed back (if provided) */
  text?: string;
  processingMs: number;
}

// ---------------------------------------------------------------------------
// Wikidata helpers
// ---------------------------------------------------------------------------

const WIKIDATA_API = "https://www.wikidata.org/w/api.php";
const CACHE_TTL_SECONDS = 60 * 60 * 24 * 7; // 7 days – entities are stable

/** Map NER labels to human-readable Wikidata type hints */
const LABEL_TYPE_MAP: Record<string, string> = {
  PER: "person",
  PERSON: "person",
  LOC: "location",
  LOCATION: "location",
  GPE: "location",
  ORG: "organization",
  ORGANIZATION: "organization",
  MISC: "item",
  EVENT: "event",
  WORK_OF_ART: "creative work",
  PRODUCT: "product",
  FAC: "facility",
  NORP: "group",
  LANGUAGE: "language",
  DATE: "time",
  TIME: "time",
  LAW: "law",
};

/** Wikidata `wbsearchentities` response shape (partial) */
interface WDSearchResult {
  id: string;
  title: string;
  label?: string;
  description?: string;
  aliases?: string[];
  url?: string;
}

interface WDSearchResponse {
  search: WDSearchResult[];
}

/**
 * Build a cache-key URL for a Wikidata lookup so the CF Cache API can
 * store/retrieve it without actually hitting the Wikidata network.
 */
function cacheKeyUrl(text: string, label: string, language: string, limit: number): string {
  const encoded = encodeURIComponent(text.toLowerCase().trim());
  return `https://wikidata-cache.internal/${language}/${label}/${limit}/${encoded}`;
}

/**
 * Query Wikidata `wbsearchentities` for one entity text.
 * Returns up to `limit` candidates ordered by Wikidata relevance score.
 */
async function searchWikidata(
  text: string,
  language: string,
  limit: number,
  typeHint: string,
): Promise<WikidataCandidate[]> {
  const params = new URLSearchParams({
    action: "wbsearchentities",
    search: text,
    language,
    uselang: language,
    limit: String(Math.min(limit * 2, 20)), // fetch extra for re-ranking
    format: "json",
    origin: "*",
  });

  const res = await fetch(`${WIKIDATA_API}?${params.toString()}`, {
    headers: {
      "User-Agent": "tflite-entity-resolver/1.0 (Cloudflare Worker; contact: metehan777@gmail.com)",
    },
  });

  if (!res.ok) {
    throw new Error(`Wikidata API error ${res.status}: ${await res.text()}`);
  }

  const data = (await res.json()) as WDSearchResponse;

  return data.search.slice(0, limit).map((item) => ({
    id: item.id,
    label: item.label ?? item.title,
    description: item.description,
    url: `https://www.wikidata.org/wiki/${item.id}`,
    aliases: item.aliases,
    typeHint,
  }));
}

/**
 * Resolve a single NER entity to Wikidata candidates.
 * Uses the CF Cache API (keyed on a synthetic URL) to avoid redundant lookups.
 */
async function resolveEntity(
  entity: NEREntity,
  language: string,
  limit: number,
  cache: Cache,
): Promise<ResolvedEntity> {
  const typeHint = LABEL_TYPE_MAP[entity.label.toUpperCase()] ?? "item";
  const key = cacheKeyUrl(entity.text, entity.label, language, limit);

  // Attempt cache hit
  const cached = await cache.match(key);
  if (cached) {
    const candidates = (await cached.json()) as WikidataCandidate[];
    return { ...entity, wikidata: candidates, cached: true };
  }

  // Cache miss – hit Wikidata
  const candidates = await searchWikidata(entity.text, language, limit, typeHint);

  // Store in CF cache
  const cacheResponse = new Response(JSON.stringify(candidates), {
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": `public, max-age=${CACHE_TTL_SECONDS}`,
    },
  });
  cache.put(key, cacheResponse); // fire-and-forget

  return { ...entity, wikidata: candidates, cached: false };
}

// ---------------------------------------------------------------------------
// Request handlers
// ---------------------------------------------------------------------------

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}

function errorResponse(message: string, status = 400): Response {
  return jsonResponse({ ok: false, error: message }, status);
}

async function handleResolve(request: Request): Promise<Response> {
  // Parse body
  let input: NERInput;
  try {
    input = (await request.json()) as NERInput;
  } catch {
    return errorResponse("Invalid JSON body");
  }

  // Validate
  if (!Array.isArray(input.entities) || input.entities.length === 0) {
    return errorResponse('Body must contain a non-empty "entities" array');
  }

  const MAX_ENTITIES = 50;
  if (input.entities.length > MAX_ENTITIES) {
    return errorResponse(`Too many entities (max ${MAX_ENTITIES} per request)`);
  }

  for (const e of input.entities) {
    if (typeof e.text !== "string" || e.text.trim() === "") {
      return errorResponse('Each entity must have a non-empty "text" string');
    }
    if (typeof e.label !== "string" || e.label.trim() === "") {
      return errorResponse('Each entity must have a non-empty "label" string');
    }
  }

  const language = typeof input.language === "string" ? input.language : "en";
  const limit = Math.max(1, Math.min(10, typeof input.limit === "number" ? input.limit : 3));

  const t0 = Date.now();

  // Open the default CF cache
  const cache = await caches.open("wikidata-entity-cache-v1");

  // Resolve all entities (parallel, bounded by CF's fetch limits)
  const resolved = await Promise.all(
    input.entities.map((entity) => resolveEntity(entity, language, limit, cache)),
  );

  const response: ResolveResponse = {
    ok: true,
    language,
    resolvedCount: resolved.filter((e) => e.wikidata.length > 0).length,
    entities: resolved,
    ...(input.text !== undefined ? { text: input.text } : {}),
    processingMs: Date.now() - t0,
  };

  return jsonResponse(response);
}

function handleDocs(): Response {
  const docs = {
    service: "TFLite NER → Wikidata Entity Resolver",
    version: "1.0.0",
    endpoints: {
      "POST /resolve": {
        description: "Resolve TFLite NER entities to Wikidata QIDs",
        body: {
          entities: "NEREntity[]  (required) — array of {text, label, start?, end?, score?}",
          text: "string       (optional) — original document text, echoed back",
          language: "string       (optional, default 'en') — BCP-47 language code",
          limit: "number       (optional, default 3, max 10) — Wikidata candidates per entity",
        },
        supportedLabels: Object.keys(LABEL_TYPE_MAP),
        example: {
          entities: [
            { text: "Barack Obama", label: "PER", score: 0.98 },
            { text: "Washington D.C.", label: "LOC", score: 0.95 },
            { text: "OpenAI", label: "ORG", score: 0.91 },
          ],
          language: "en",
          limit: 2,
        },
      },
      "GET /health": { description: "Liveness check" },
      "GET /": { description: "This documentation" },
    },
    limits: {
      maxEntitiesPerRequest: 50,
      maxCandidatesPerEntity: 10,
      cacheTtl: "7 days",
    },
  };
  return jsonResponse(docs);
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export default {
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const { pathname, method } = { pathname: url.pathname, method: request.method };

    // CORS preflight
    if (method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
          "Access-Control-Max-Age": "86400",
        },
      });
    }

    // Route
    if (pathname === "/resolve" && method === "POST") {
      return handleResolve(request);
    }

    if (pathname === "/health" && method === "GET") {
      return jsonResponse({ ok: true, timestamp: new Date().toISOString() });
    }

    if (pathname === "/" && method === "GET") {
      return handleDocs();
    }

    return errorResponse(`Cannot ${method} ${pathname}`, 404);
  },
} satisfies ExportedHandler;
