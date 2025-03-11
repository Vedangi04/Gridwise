import React, { useState } from 'react';
import './Sidebar.css';

const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false);

  const toggleCollapsed = () => {
    setCollapsed(!collapsed);
  };

  return (
    <div className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <button className="toggle-button" onClick={toggleCollapsed}>
        {collapsed ? '>>' : '<<'}
      </button>
      <div className="sidebar-content">
        <div className="section">
          <h2>Total Consumption</h2>
          <ul>
            <li>Overall</li>
            <li>Pie chart</li>
            <li>Bar Graph</li>
          </ul>
        </div>
        <div className="section">
          <h2>Wind Power</h2>
          <ul>
            <li>Prediction</li>
            <li>Average Power</li>
            <li>Max Power</li>
          </ul>
        </div>
        <div className="section">
          <h2>Solar Power</h2>
          <ul>
          <li>Prediction</li>
            <li>Average Power</li>
            <li>Max Power</li>
          </ul>
        </div>
        <div className="section">
          <h2>Machine Consumption</h2>
          <ul>
            <li>Consumption</li>
            <li>Temperature</li>
            <li>Anomaly</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
