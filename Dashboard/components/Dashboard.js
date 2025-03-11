// import React, { useState, useEffect } from 'react';
// import axios from 'axios';
// import LineGraph from './LineGraph';

// const API_URL = 'http://localhost:3001';
// const Dashboard = () => {
//   const [showSolarTable, setShowSolarTable] = useState(false);
//   const [showWindTable, setShowWindTable] = useState(false);
//   const [windData, setWindData] = useState([]);

//   const handleSolarButtonClick = () => {
//     setShowSolarTable(true);
//     // setShowWindTable(false);
//   };

//   const handleWindButtonClick = async() => {
//     await getWindPrediction();
//   };

//   const getWindPrediction = async () => {
//     try {
//       console.log('Fetching wind prediction data...');
//       const category = 'windprediction';
//       const response = await axios.post(`${API_URL}/api/getprediction`, JSON.stringify({ category }), {
//         method: 'POST',
//         headers: {
//           'content-type': 'application/json',
//         },
//       });
//       const responseData = response.data.apidata;
//       console.log('Fetched data:', responseData);
//       setWindData(responseData);
//       setShowWindTable(true);
//     } catch (error) {
//       console.error('Error fetching wind prediction data:', error);
//     }
//   };

//   useEffect(() => {
//     if(windData.length !== 0)
//       setShowWindTable(true);
//     // setShowSolarTable(false);
//   }, [windData]);

//   return (
//     <div>
//       <h1>GridWise</h1>
//       <div>
//         <button onClick={handleSolarButtonClick}>Solar Generation</button>
//         <button onClick={handleWindButtonClick}>Wind Generation</button>
//       </div>
//       {showSolarTable && (
//         <div>
//           <h2>Solar Power Generation Data</h2>
//           {/* <CSVDisplay filePath="/solar_prediction_data.csv" /> */}
//         </div>
//       )}
//       {showWindTable && (
//         <div>
//           <div>
//             <h2>Power Data Visualization</h2>
//             <LineGraph data={windData} />
//           </div>
//           <div>
//             <h2>Wind Power Generation Data</h2>
//             {/* <CSVDisplay filePath="/wind_prediction_data.csv" /> */}
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Dashboard;
