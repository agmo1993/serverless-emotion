import React, { Component } from 'react';
import { Navigation } from './components/navigation.component';
import { MainContent } from './components/main-content.component';
import './index.css';


class App extends Component{
  constructor() {
    super();
    this.state = {
       data: []
    }
 }
  render(){
    return(
      <div>
        <MainContent />
     
        <div class="section">
        <div class="cards">
          <h2>Precision</h2>
          <button>See More</button>
        </div>
        <div class="cards">
          <h2>Uptime</h2>
          <button>See More</button>
        </div>
        <div class="cards">
          <h2>Maximum Load</h2>
          <button>See More</button>
        </div>
        </div>
        <div class="footer"><h3>Contact</h3></div>
      </div>
      
    )
  }
}

export default App;
