// Using the document$ observable from mkdocs-material to get notified of page "reload" also if using `navigation.instant` (SSA)
document$.subscribe(function() {
  // First check if the page contains a notebook-related class
  if (document.querySelector('.jp-Notebook')) {
    document.querySelector("div.md-sidebar.md-sidebar--secondary").remove();
 //   document.querySelector("div.md-sidebar.md-sidebar--primary").remove();
  }
});


