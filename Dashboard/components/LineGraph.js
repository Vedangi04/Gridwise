// import React from 'react';
// import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
// import { format } from 'date-fns';

// function LineGraph({ data }) {
//   console.log('CCCCCCCCCCCC ', data);
//   // Check if data is an array
//   if (!Array.isArray(data)) {
//     return <div>Error: Data is not an array</div>;
//   }

//   // Extract Time, Actual Power, and Predicted Power from the data array
//   const parsedData = data.map(item => ({
//     ...item,
//     Time: convertDateFormat(item.Time)
//   }));
//   console.log('PPPPPPPPPPPPPPPPP', parsedData.map(item => item.Time));
//   const labels = parsedData.map(item => item.Time);
//   console.log('KKKKKK', labels);
//   const actualPower = parsedData.map(item => item['Actual Power']);
//   console.log('KKKKKK', actualPower);
//   const predictedPower = parsedData.map(item => item['Predicted Power']);
//   console.log('KKKKKK', predictedPower);

//   return (
//     <div>
//       <h2>Line Graph</h2>
//       <LineChart
//         width={600}
//         height={300}
//         data={parsedData}
//         margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
//       >
//         <CartesianGrid strokeDasharray="3 3" />
//         <XAxis dataKey="Time" />
//         <YAxis />
//         <Tooltip />
//         <Legend />
//         <Line
//           type="monotone"
//           dataKey="Actual Power"
//           stroke="rgb(75, 192, 192)"
//           dot={false}
//         />
//         <Line
//           type="monotone"
//           dataKey="Predicted Power"
//           stroke="rgb(255, 99, 132)"
//           dot={false}
//         />
//       </LineChart>
//     </div>
//   );
// }

// function convertDateFormat(inputDate) {
//   // Split the input date string into components
//   const parsedDate = new Date(inputDate);

//   // Format the date in the desired format (YYYY-MM-DDTHH:mm:ss)
//   const formattedDate = format(parsedDate, "yyyy-MM-dd'T'HH:mm:ss");

//   return formattedDate;
// }

// export default LineGraph;
