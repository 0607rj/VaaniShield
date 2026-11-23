// Shared header, footer and small interactions for every VaaniShield page.
(function () {
  const LOGO = `<svg viewBox="0 0 40 40" aria-hidden="true"><path d="M20 3 34 8.5V19c0 8.6-6 15.4-14 18-8-2.6-14-9.4-14-18V8.5Z" fill="#3b3bd6"/><g fill="#fff"><rect x="12" y="17" width="2.6" height="7" rx="1.3"/><rect x="16.3" y="13.5" width="2.6" height="14" rx="1.3"/><rect x="20.6" y="11" width="2.6" height="19" rx="1.3"/><rect x="24.9" y="15" width="2.6" height="11" rx="1.3"/></g><circle cx="31" cy="9" r="4" fill="#ff8a1f"/></svg>`;
  const PAGES = [["/", "Home"], ["/features", "Features"], ["/how-it-works", "How it works"], ["/extension", "Extension"], ["/faq", "FAQ"], ["/contact", "Contact"]];
  const here = location.pathname.replace(/\.html$/, "").replace(/\/$/, "") || "/";

  const header = document.createElement("header");
  header.className = "nav";
  header.innerHTML = `<div class="wrap">
      <a class="logo" href="/">${LOGO}<span>Vaani<b>Shield</b></span></a>
      <nav class="links" id="links">
        ${PAGES.map(([h, t]) => `<a href="${h}" class="${(h === "/" ? here === "/" : here.startsWith(h)) ? "active" : ""}">${t}</a>`).join("")}
        <a class="btn small" href="/try">Try it free <span class="arrow">→</span></a>
      </nav>
      <button class="burger" aria-label="Menu" onclick="document.getElementById('links').classList.toggle('open')"><span></span><span></span><span></span></button>
    </div>`;
  document.body.prepend(header);

  const footer = document.createElement("footer");
  footer.innerHTML = `<div class="wrap">
      <div>
        <a class="logo" href="/">${LOGO}<span>Vaani<b>Shield</b></span></a>
        <p style="margin-top:14px;max-width:30ch">Know when a voice is real — before you trust it.</p>
      </div>
      <nav>
        <div><strong>Product</strong><a href="/features">Features</a><a href="/try">Try it</a><a href="/extension">Browser extension</a></div>
        <div><strong>Learn</strong><a href="/how-it-works">How it works</a><a href="/faq">FAQ</a></div>
        <div><strong>Talk to us</strong><a href="/contact">Contact</a></div>
      </nav>
      <div class="fine">© ${new Date().getFullYear()} VaaniShield · Research prototype. Results are a model estimate, not forensic proof.</div>
    </div>`;
  document.body.append(footer);

  addEventListener("scroll", () => header.classList.toggle("scrolled", scrollY > 8), { passive: true });

  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.15 });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

  // live accuracy numbers from the server, wherever [data-metric] is used
  const els = document.querySelectorAll("[data-metric]");
  if (els.length) fetch("/api/models").then((r) => r.json()).then((models) => {
    const by = Object.fromEntries(models.map((m) => [m.key, m.metrics]));
    els.forEach((el) => {
      const [model, split, field] = el.dataset.metric.split(".");
      const v = by[model]?.[split]?.[field] ?? by[model]?.dev?.[field];
      if (v != null) el.textContent = typeof v === "number" ? (field.includes("percent") ? v + "%" : v.toLocaleString()) : v;
    });
  }).catch(() => {});
})();
