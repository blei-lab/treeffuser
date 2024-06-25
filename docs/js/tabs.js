// script.js

// Parse tabcontents and in each cell, make X.xx ± Y.yy to have Y.yy smaller font size
// This is a hacky way to do it, but it works

document.addEventListener("DOMContentLoaded", function() {
    var tabcontents = document.getElementsByClassName("tabcontent");
    for (var i = 0; i < tabcontents.length; i++) {
      var tabcontent = tabcontents[i];
      var cells = tabcontent.getElementsByTagName("td");
      for (var j = 0; j < cells.length; j++) {
        var cell = cells[j];
        var text = cell.innerHTML;
        var match = text.match(/(\d+\.\d+)±(\d+\.\d+)/);
        if (match) {
          var mean = match[1];
          mean = parseFloat(mean).toFixed(1);
          var sd = match[2];
          sd = parseFloat(sd).toFixed(1);
          cell.innerHTML = mean + "&nbsp;±<span style='font-size: 0.8em;'>" + sd + "</span>";
        }
      }
    }
    });

function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
      tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
      tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
  }

  // Get the element with id="defaultOpen" and click on it
  document.addEventListener("DOMContentLoaded", function() {
    document.querySelector('.tablinks').click();
  });
