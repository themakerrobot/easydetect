// Light or dark: the visitor's choice if made, else the system setting.
(() => {
  const KEY = "hub.theme";
  const root = document.documentElement;
  const sys = matchMedia("(prefers-color-scheme: dark)");
  const saved = () => { try { return localStorage.getItem(KEY); } catch { return null; } };
  const set = (t) => { root.dataset.theme = t; };
  set(saved() || (sys.matches ? "dark" : "light"));
  sys.addEventListener("change", (e) => { if (!saved()) set(e.matches ? "dark" : "light"); });
  const label = (b) => {
    const dark = root.dataset.theme === "dark", ko = root.lang !== "en";
    b.title = b.ariaLabel = dark ? (ko ? "밝게 보기" : "Light theme") : (ko ? "어둡게 보기" : "Dark theme");
  };
  document.addEventListener("DOMContentLoaded", () => {
    // the version badge: easydetect's latest release on PyPI
    const ver = document.querySelector(".ver");
    if (ver) fetch("https://pypi.org/pypi/easydetect/json")
      .then((r) => r.ok ? r.json() : null)
      .then((j) => { if (j && j.info && j.info.version) ver.textContent = "v" + j.info.version; })
      .catch(() => {});
    const b = document.getElementById("theme");
    if (!b) return;
    label(b);
    new MutationObserver(() => label(b)).observe(root, { attributes: true, attributeFilter: ["lang", "data-theme"] });
    b.addEventListener("click", () => {
      const t = root.dataset.theme === "dark" ? "light" : "dark";
      set(t);
      try { localStorage.setItem(KEY, t); } catch {}
    });
  });
})();
