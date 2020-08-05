import React from 'react';
import {Navigation} from './navigation.component'
import {DynamicContent} from './dynamic-content.component'
import './components.style.css';

export const MainContent = props => (
    <header>
        <img class="header-image" src="../images/smile@2x.jpg" alt="picture of happy person"/>
        <a href="#" class="header-logo">EMOTION LENS</a>
        <Navigation/>
        <DynamicContent/>
    </header>
);