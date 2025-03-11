import React from 'react';
import './Topbar.css'; 

const Topbar = () => {
  return (
    <div className="topbar">
      <div className="search-bar">
        <input type="text" placeholder="Search..." />
        <button><i className="fas fa-search"></i></button>
    </div>
        <div className="icons">
        </div>
    </div>
  );
};

export default Topbar;
