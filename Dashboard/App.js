import React, { useState } from 'react';
import Dashboard from './components/Dashboard';
import Sidebar from './components/Sidebar';
import Topbar from './components/Topbar';
import './App.css'; 

function App() {
  const [showDashboard, setShowDashboard] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);
  const [showTopbar, setShowTopbar] = useState(false);

  const toggleDashboard = () => {
    setShowDashboard(!showDashboard);
  };

  const toggleTopbar = () => {
    setShowTopbar(!showTopbar);
  };

  const toggleSidebar = () => {
    setShowSidebar(!showSidebar);
  };

  return (
    <div className="App">
      {showDashboard && <Dashboard />}
      <button onClick={toggleDashboard}>Toggle Dashboard</button>

      {showSidebar && <Sidebar />}
      <button onClick={toggleSidebar}>Toggle Sidebar</button>

      {showTopbar && <Topbar />}
      <button onClick={toggleTopbar}>Toggle Topbar</button>
    </div>
  );
}

export default App;
