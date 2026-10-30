const $ = (sel, root = document) => root.querySelector(sel);

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const fmtTime = (iso) =>
  new Date(iso).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", hour12: false });

async function api(method, path, body) {
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Error("서버에 연결할 수 없어요");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function poll(fn, ms = 2000) {
  const tick = async () => {
    const conn = $("#conn");
    try {
      await fn();
      if (conn) conn.classList.add("hidden");
    } catch (e) {
      console.error(e);
      if (conn) conn.classList.remove("hidden");
    }
    setTimeout(tick, typeof ms === "function" ? ms() : ms);
  };
  tick();
}
