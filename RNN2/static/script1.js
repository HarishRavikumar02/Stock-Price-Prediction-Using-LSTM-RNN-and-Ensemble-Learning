  const formOpenBtn = document.querySelector("#form_open"),
  home = document.querySelector(".home"),
  formContainer = document.querySelector(".form_container"),
  formCloseBtn = document.querySelector(".form_close"),
  signupBtn = document.querySelector("#signup"),
  loginBtn = document.querySelector("#login"),
  pwShowHide = document.querySelectorAll(".pw_hide");

  formOpenBtn.addEventListener("click", () => home.classList.add("show"));
  formCloseBtn.addEventListener("click", () => home.classList.remove("show"));

  pwShowHide.forEach(icon =>{
   icon.addEventListener("click", () => {
   let getPwInput = icon.parentElement.querySelector("input")   
   console.log(getPwInput);
   if(getPwInput.type === "password"){
        getPwInput.type = "text";
        icon.classList.replace("uil-eye-slash", "uil-eye")
    }else{
        getPwInput.type = "password";
        icon.classList.replace("uil-eye", "uil-eye-slash")
         }
    })    
  });

 signupBtn.addEventListener("click", (e) => {
   e.preventDefault();
   formContainer.classList.add("active");
 })

 loginBtn.addEventListener("click", (e) => {
   e.preventDefault();
   formContainer.classList.remove("active");
 })

  const apiKey = 'ZAC6RA387SU7FN0B';
  const apple = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=ZAC6RA387SU7FN0B`;
  const google = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=GOOGL&apikey=ZAC6RA387SU7FN0B`;
  const amazon = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AMZN&apikey=ZAC6RA387SU7FN0B`;
  const microsoft = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=MSFT&apikey=ZAC6RA387SU7FN0B`;
  const tesla = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=TSLA&apikey=ZAC6RA387SU7FN0B`;
        
  function updateStockPrice() {
    fetch(apple)
      .then(response => response.json())
      .then(data => {
        const price = data['Global Quote']['05. price'];
        document.getElementById('apple').innerHTML = `${price}`;
      })
      .catch(error => {
        console.error(error);
      });

      fetch(google)
      .then(response => response.json())
      .then(data => {
        const price = data['Global Quote']['05. price'];
        document.getElementById('google').innerHTML = `${price}`;
      })
      .catch(error => {
        console.error(error);
      });

      fetch(amazon)
      .then(response => response.json())
      .then(data => {
        const price = data['Global Quote']['05. price'];
        document.getElementById('amazon').innerHTML = `${price}`;
      })
      .catch(error => {
        console.error(error);
      });

      fetch(microsoft)
      .then(response => response.json())
      .then(data => {
        const price = data['Global Quote']['05. price'];
        document.getElementById('microsoft').innerHTML = `${price}`;
      })
      .catch(error => {
        console.error(error);
      });

      fetch(tesla)
      .then(response => response.json())
      .then(data => {
        const price = data['Global Quote']['05. price'];
        document.getElementById('tesla').innerHTML = `${price}`;
      })
      .catch(error => {
        console.error(error);
      });
  }

  updateStockPrice(); // call once immediately
  setInterval(updateStockPrice, 1000); // update second minute          