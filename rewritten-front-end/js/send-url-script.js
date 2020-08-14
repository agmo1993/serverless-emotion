function loadDoc(cFunction) {
    console.log("trigger");
    var xhttp = new XMLHttpRequest();
    xhttp.open("POST", "http://45.113.235.20:8080/function/emotion-model-http", true);
    xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhttp.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
        cFunction(this);
      }
    };
    var inputVal = document.getElementById("myInput").value;
    xhttp.send(inputVal);
  }
  
  function myFunction(xhttp) {
    document.getElementById("demo").innerHTML =
    xhttp.responseText;
    alert(xhttp.responseText);
  }