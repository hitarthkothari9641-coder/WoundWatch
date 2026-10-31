// Edge gateway: static UI comes from Workers assets; /api, /predict go to the HF Space with a shared secret.
const SECURITY = {
  "Content-Security-Policy": "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
  "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Referrer-Policy": "no-referrer",
  "Permissions-Policy": "camera=(self), microphone=(), geolocation=(), payment=()",
  "Cross-Origin-Opener-Policy": "same-origin",
};
const secure = (res) => {
  const r = new Response(res.body, res);
  for (const [k, v] of Object.entries(SECURITY)) r.headers.set(k, v);
  r.headers.delete("server");
  return r;
};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const proxied = /^\/(api\/|predict$|health$|live$)/.test(url.pathname);
    if (!proxied) return secure(await env.ASSETS.fetch(request));

    if (env.LIMITER) {
      const key = request.headers.get("cf-connecting-ip") || "unknown";
      const { success } = await env.LIMITER.limit({ key });
      if (!success) return secure(new Response(JSON.stringify({ detail: "Too many requests" }), { status: 429, headers: { "content-type": "application/json", "retry-after": "60" } }));
    }
    const target = new URL(url.pathname + url.search, `https://${env.BACKEND_HOST}`);
    const headers = new Headers(request.headers);
    headers.set("host", env.BACKEND_HOST);
    headers.set("x-origin-secret", env.ORIGIN_SECRET);
    if (headers.has("origin")) headers.set("origin", `https://${env.BACKEND_HOST}`); // backend CSRF origin check is host-relative
    headers.delete("cf-connecting-ip");
    const upstream = await fetch(new Request(target, { method: request.method, headers, body: request.body, redirect: "manual" }));
    return secure(upstream);
  },
};
