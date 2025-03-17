import React, { useState, useEffect } from 'react';
import axios from 'axios';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  Tooltip,
  Label,
} from 'recharts';

// Chakra imports
import {
  Box,
  Flex,
  Grid,
  SimpleGrid,
} from "@chakra-ui/react";


// Custom components
import MiniStatistics from "components/card/MiniStatistics";
// Assets

const API_URL = 'http://localhost:3001';
export default function WindPowerPrediction() {
  const [selectedDate, setSelectedDate] = useState(new Date("01-02-2022"));
  const [windData, setWindData] = useState([]);
  const [averageActualPower, setAverageActualPower] = useState(null);
  const [maximumActualPower, setMaximumActualPower] = useState(null);
  const [totalWindPowerGenerated, setTotalWindPowerGenerated] = useState(null);

  
  useEffect(() => {
    const getWindPrediction = async () => {
      try {
        console.log('Fetching wind prediction data...');
        const filterCriteria = formatDate(selectedDate);
        const response = await axios.post(`${API_URL}/api/getprediction`, JSON.stringify({ filterCriteria }), {
          headers: {
            'Content-Type': 'application/json',
          },
        });
        const responseData = response.data.apidata.map(item => ({
          ...item,
          time: item.time.trim().split(' ')[1], // Extract time after space
        }));
        console.log('Fetched data:', responseData);
        setWindData(responseData);
        setAverageActualPower(response.data.averageActualPower);
        setMaximumActualPower(response.data.maximumActualPower);
        setTotalWindPowerGenerated(response.data.totalWindPowerGenerated);
      } catch (error) {
        console.error('Error fetching wind prediction data:', error);
      }
    };
    getWindPrediction();
  }, [selectedDate]);

  const handleDateChange = date => {
    setSelectedDate(date);
  };

  const formatDate = date => {
    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    return `${day}-${month}-${year}`;
  };

  return (
    <Box pt={{ base: "180px", md: "80px", xl: "80px" }}>
      {/* Main Fields */}
      <Grid
        mb='20px'
        gridTemplateColumns={{ xl: "repeat(3, 1fr)", "2xl": "1fr 0.46fr" }}
        gap={{ base: "20px", xl: "20px" }}
        display={{ base: "block", xl: "grid" }}>
        <Flex
          flexDirection='column'
          gridArea={{ xl: "1 / 1 / 2 / 3", "2xl": "1 / 1 / 2 / 2" }}>
          <Flex direction='column'>            
            {averageActualPower !== null && maximumActualPower !== null && totalWindPowerGenerated !== null && (
            <SimpleGrid
              columns={{ base: 1, md: 3 }}
              gap='20px'
              mb={{ base: "20px", xl: "0px" }}>
              {averageActualPower !== null && (
                <MiniStatistics
                  name='Average Actual Power'
                  value={`${averageActualPower.toFixed(2)} kW`}
                />
              )} 
              {maximumActualPower !== null && (
                <>
                <MiniStatistics
                  name='Maximum Actual Power'
                  value={`${maximumActualPower.toFixed(2)} kW`}
                />
                <MiniStatistics
                  name='Total Power Generated'
                  value={`${totalWindPowerGenerated.toFixed(2)} kW`}
                /> 
              </>
              )} 
            </SimpleGrid>
            )} 
          </Flex>
        </Flex>        
      </Grid>
      <SimpleGrid columns={{ base: 1, md: 2, xl: 2 }} gap='20px' mb='20px'>
        <h1 className="chart-heading">Wind Power Prediction Chart</h1>
        <SimpleGrid columns={{ base: 1, md: 1, xl: 1 }} gap='20px'>
          <DatePicker selected={selectedDate} onChange={handleDateChange} dateFormat="dd-MM-yyyy" />
        </SimpleGrid>        
      </SimpleGrid>
      <ResponsiveContainer width="100%" aspect={3}>
        <LineChart data={windData} width={500} height={300} margin={{ top: 5, right: 100, left: 30, bottom: 5 }}>
          <CartesianGrid />
          <XAxis dataKey="time" interval={'preserveStartEnd'} >
          <Label value="Time (hr)" offset={-5} position="insideBottom" />
          </XAxis>
          <YAxis >
          <Label value="Power (kWh)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />
          </YAxis>
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="ActualPower" stroke='green' activeDot={{ r: 8 }} />
          <Line type="monotone" dataKey="PredictedPower" stroke='blue' activeDot={{ r: 8 }} />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
}
