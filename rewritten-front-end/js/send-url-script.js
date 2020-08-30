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
  
  document.getElementById("components_result").style.display = "none";
  document.getElementById("cross_result").style.display = "block";
  
  document.getElementById("demo").innerHTML = xhttp.responseText;
  console.log("Completed")

    // document.getElementById("span_result").style.display = "none";
    // document.getElementById("icon_result").style.display = "none";
    // document.getElementById("cross_result").style.display = "block";
  
    
    //alert(xhttp.responseText);
}

function returnElements() {
  document.getElementById("cross_result").style.display = "none";
  document.getElementById("components_result").style.display = "block";
  document.getElementById("demo").innerHTML = "";
}