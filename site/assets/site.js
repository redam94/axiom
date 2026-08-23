/* Theme toggle, copy buttons, and the API filter. Nothing here is required to
   read the page — it all degrades to a working static document. */

(function () {
  "use strict";

  /* -- theme --------------------------------------------------------------- */

  var root = document.documentElement;
  var btn = document.querySelector(".theme");

  function current() {
    var set = root.getAttribute("data-theme");
    if (set) return set;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  if (btn) {
    btn.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("axiom-theme", next); } catch (e) { /* private mode */ }
      if (window.AxiomCharts) window.AxiomCharts.redraw();
    });
  }

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
    if (!root.getAttribute("data-theme") && window.AxiomCharts) window.AxiomCharts.redraw();
  });

  /* -- copy buttons -------------------------------------------------------- */

  document.querySelectorAll(".code").forEach(function (block) {
    var btn = block.querySelector(".copy");
    var pre = block.querySelector("pre");
    if (!btn || !pre) return;
    btn.addEventListener("click", function () {
      navigator.clipboard.writeText(pre.innerText).then(function () {
        var was = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = was; }, 1400);
      }, function () {
        btn.textContent = "Press ⌘C";
      });
    });
  });

  /* -- API filter ---------------------------------------------------------- */

  var search = document.querySelector("#api-q");
  if (search) {
    var packages = Array.prototype.slice.call(document.querySelectorAll(".api-pkg"));
    var count = document.querySelector("#api-count");
    var total = document.querySelectorAll(".sym").length;

    var run = function () {
      var q = search.value.trim().toLowerCase();
      var shown = 0;
      packages.forEach(function (pkg) {
        var any = false;
        var pkgName = pkg.dataset.pkg;
        var pkgMatches = q && pkgName.indexOf(q) !== -1;
        pkg.querySelectorAll(".sym").forEach(function (sym) {
          /* data-search is the name plus its one-line docstring. Deliberately not
             textContent: that now includes the usage snippet, so every symbol
             would match a common word like "graph" through someone else's code. */
          var hay = sym.dataset.search || sym.textContent.toLowerCase();
          var hit = !q || pkgMatches || hay.indexOf(q) !== -1;
          sym.hidden = !hit;
          sym.classList.toggle("hit", !!q && hit && !pkgMatches);
          if (hit) { any = true; shown++; }
        });
        pkg.hidden = !any;
      });
      count.textContent = q
        ? shown + " of " + total + " symbols match “" + search.value.trim() + "”"
        : total + " public symbols across " + packages.length +
          " subpackages — click one for its signature and a real call";
    };

    search.addEventListener("input", run);
    run();

    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && document.activeElement !== search) {
        e.preventDefault();
        search.focus();
        search.select();
      }
    });
  }
})();
