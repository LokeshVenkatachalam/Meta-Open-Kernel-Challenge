(() => {
  "use strict";

  document.documentElement.classList.add("has-js");

  const ledger = document.querySelector("[data-experiment-ledger]");
  if (!ledger) {
    return;
  }

  const form = ledger.querySelector("[data-filter-form]");
  const problem = ledger.querySelector('[data-filter="problem"]');
  const mode = ledger.querySelector('[data-filter="mode"]');
  const outcome = ledger.querySelector('[data-filter="outcome"]');
  const summary = ledger.querySelector("[data-filter-summary]");
  const noMatch = ledger.querySelector("[data-no-match]");
  const rows = Array.from(ledger.querySelectorAll("[data-run-row]"));

  if (!form || !problem || !mode || !outcome || !summary || !noMatch || rows.length === 0) {
    const notice = document.createElement("p");
    notice.className = "filter-notice";
    notice.setAttribute("role", "alert");
    notice.textContent = "Experiment filters could not be initialized. Reload the page or ";
    const download = document.createElement("a");
    download.href = "data/runs.json";
    download.textContent = "download the run dataset.";
    notice.append(download);
    ledger.prepend(notice);
    form?.querySelectorAll("select, button").forEach((control) => {
      control.disabled = true;
    });
    console.error("Experiment filters are missing required controls or run rows.");
    return;
  }

  const invalidFilters = [];
  const selectValueIfValid = (select, value) => {
    if (!value) {
      return;
    }
    const valid = Array.from(select.options).some((option) => option.value === value);
    if (valid) {
      select.value = value;
    } else {
      invalidFilters.push(select.name);
    }
  };

  const params = new URLSearchParams(window.location.search);
  selectValueIfValid(problem, params.get("problem"));
  selectValueIfValid(mode, params.get("mode"));
  selectValueIfValid(outcome, params.get("outcome"));

  let queryNotice = null;
  if (invalidFilters.length > 0) {
    queryNotice = document.createElement("p");
    queryNotice.className = "filter-notice";
    queryNotice.setAttribute("role", "status");
    queryNotice.textContent = `Invalid URL filters were ignored: ${invalidFilters.join(", ")}. The remaining valid filters are shown.`;
    form.before(queryNotice);
    console.warn(`Ignored invalid experiment filters: ${invalidFilters.join(", ")}`);
  }

  const applyFilters = () => {
    let visible = 0;
    rows.forEach((row) => {
      const problemMatch = problem.value === "all" || row.dataset.problem === problem.value;
      const modeMatch = mode.value === "all" || row.dataset.mode === mode.value;
      const outcomeMatch = outcome.value === "all" || row.dataset.outcome === outcome.value;
      const show = problemMatch && modeMatch && outcomeMatch;
      row.hidden = !show;
      if (show) {
        visible += 1;
      }
    });

    summary.textContent = visible === rows.length
      ? `Showing all ${rows.length} attempts.`
      : `Showing ${visible} of ${rows.length} attempts.`;
    noMatch.hidden = visible !== 0;
  };

  form.addEventListener("change", () => {
    queryNotice?.remove();
    applyFilters();
  });
  form.addEventListener("reset", () => window.setTimeout(() => {
    queryNotice?.remove();
    applyFilters();
  }, 0));
  applyFilters();
})();
