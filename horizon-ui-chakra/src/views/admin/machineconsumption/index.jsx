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
export default function MachineConsumption() {
  const [selectedDate, setSelectedDate] = useState(new Date("01-01-2023"));
  const [consumptionData, setConsumptionData] = useState([]);
  const [totalPowerConsumption, setTotalPowerConsumption] = useState(null);
  const [anomalyDetection, setAnomalyDetection] = useState(false);
  useEffect(() => {
    const getMachineConsumption = async () => {
      try {
        console.log('Fetching machine consumption data...');
        const filterCriteria = formatDate(selectedDate);
        const response = await axios.post(`${API_URL}/api/getconsumption`, JSON.stringify({ filterCriteria }), {
          headers: {
            'Content-Type': 'application/json',
          },
        });
        const responseData = response.data.apidata.map(item => ({
          ...item,
          time: item.time.trim().split(' ')[1], // Extract time after space
        }));
        setConsumptionData(responseData);
        setTotalPowerConsumption(response.data.totalPowerConsumption);
        setAnomalyDetection(response.data.anomalies);
      } catch (error) {
        console.error('Error fetching machine consumption data:', error);
      }
    };
    getMachineConsumption();
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
            {totalPowerConsumption !== null && (
            <SimpleGrid
              columns={{ base: 1, md: 3 }}
              gap='20px'
              mb={{ base: "20px", xl: "0px" }}>
              {totalPowerConsumption !== null && (
                <MiniStatistics
                  name='Total Power Consumption'
                  value={`${totalPowerConsumption.toFixed(2)} kWh`}
                />
              )} 
              {totalPowerConsumption !== null && (
                <MiniStatistics
                name='Anomaly Detection'
                value={anomalyDetection}
              />
              )} 
            </SimpleGrid>
            )} 
          </Flex>
        </Flex>        
      </Grid>
      <SimpleGrid columns={{ base: 1, md: 2, xl: 2 }} gap='20px' mb='20px'>
        <h1 className="chart-heading">Machine Consumption Chart</h1>
        <SimpleGrid columns={{ base: 1, md: 1, xl: 1 }} gap='20px'>
          <DatePicker selected={selectedDate} onChange={handleDateChange} dateFormat="dd-MM-yyyy" />
        </SimpleGrid>        
      </SimpleGrid>
      <ResponsiveContainer width="100%" aspect={3}>
        <LineChart width={500} height={300} data={consumptionData} margin={{ top: 5, right: 100, left: 30, bottom: 5 }}>
          <CartesianGrid />
          <XAxis dataKey="time" interval={'preserveStartEnd'}>
          <Label value="Time (hr)" offset={-5} position="insideBottom" />
          </XAxis>
          <YAxis>
          <Label value="Energy Consumed (kWh)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />
          </YAxis>
          <Tooltip />
          <Legend />
          {Array.from({ length: 5 }, (_, index) => (
            <Line
              key={`Machine_${index + 1}_Energy`}
              type="monotone"
              dataKey={`Machine_${index + 1} Energy Consumed (kWh)`}
              stroke={index === 0 ? 'green' : index === 1 ? 'blue' : index === 2 ? 'red' : index === 3 ? 'yellow' : 'orange'}
              activeDot={{ r: 8 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <ResponsiveContainer width="100%" aspect={3}>
        <LineChart width={500} height={300} data={consumptionData} margin={{ top: 5, right: 100, left: 30, bottom: 5 }}>
          <CartesianGrid />
          <XAxis dataKey="time" interval={'preserveStartEnd'}>
          <Label value="Time (hr)" offset={-5} position="insideBottom" />
          </XAxis>
          <YAxis>
          <Label value="Temperature (°C)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />
          </YAxis>
          <Tooltip />
          <Legend />
          {Array.from({ length: 5 }, (_, index) => (
            <Line
              key={`Machine_${index + 1}_Temperature`}
              type="monotone"
              dataKey={`Machine_${index + 1} Temperature (C)`}
              stroke={index === 0 ? 'green' : index === 1 ? 'blue' : index === 2 ? 'red' : index === 3 ? 'yellow' : 'orange'}
              activeDot={{ r: 8 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
}
