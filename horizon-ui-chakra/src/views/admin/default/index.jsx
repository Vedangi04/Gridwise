import {
  Box,
  SimpleGrid,
  Button
} from "@chakra-ui/react";
import axios from 'axios';
import DatePicker from 'react-datepicker';
import MiniStatistics from "components/card/MiniStatistics";
import React, { useState, useEffect } from 'react';
import PieCard from "views/admin/default/components/PieCard";

const API_URL = 'http://localhost:3001';
export default function UserReports() {

  const [selectedDate, setSelectedDate] = useState(new Date("01-02-2022"));
  const [combinedTotalPower, setCombinedTotalPower] = useState([]);
  const [totalWindPowerGenerated, setTotalWindPowerGenerated] = useState(null);
  const [totalSolarPowerGenerated, setTotalSolarPowerGenerated] = useState(null);
  const [pieChartData, setPieChartData] = useState([]);
  useEffect(() => {
    const fetchData  = async () => {
      try {
        const filterCriteria = formatDate(selectedDate);
        const response = await axios.post(`${API_URL}/api/getcombinedtotalpower`, JSON.stringify({ filterCriteria }), {
          headers: {
            'Content-Type': 'application/json',
          },
        });
        setCombinedTotalPower(response.data.combinedTotalPower);
        setTotalWindPowerGenerated(response.data.totalWindPowerGenerated);
        setTotalSolarPowerGenerated(response.data.totalSolarPowerGenerated);
        const solarPercentage = Math.round(response.data.totalSolarPowerGenerated / response.data.combinedTotalPower * 100);
        console.log('ssssssssssssssss', solarPercentage);
        const windPercentage = Math.round(response.data.totalWindPowerGenerated / response.data.combinedTotalPower * 100);
        console.log('wwwwwwwww', windPercentage);
        setPieChartData([solarPercentage, windPercentage]);
      } catch (error) {
        console.error('Error fetching wind prediction data:', error);
      }
    };
    fetchData();
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

  const handleGenerateReport = async () => {
    try {
      const filterCriteria = formatDate(selectedDate);
      const response = await axios.post(`${API_URL}/api/generateReport`, { filterCriteria }, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'report.xlsx');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Error generating report:', error);
    }
  };


  return (
    <Box pt={{ base: "130px", md: "80px", xl: "80px" }}>
        <SimpleGrid
        columns={{ base: 1, md: 1, lg: 4, "2xl": 6 }}
        gap='20px'
        mb='20px'>
        <SimpleGrid columns={{ base: 1, md: 2, xl: 1 }} gap='20px'>
          <p>Date : <DatePicker selected={selectedDate} onChange={handleDateChange} dateFormat="dd-MM-yyyy" /></p>
          <Button onClick={handleGenerateReport}>Generate Report</Button>
        </SimpleGrid> 
        {combinedTotalPower !== null && (
          <MiniStatistics
          name='Total Generation'
          value={`${combinedTotalPower} kWh`}
        />
        
        )}
        {totalWindPowerGenerated !== null && (
        <MiniStatistics
          name='Wind Contribution'
          value={`${totalWindPowerGenerated.toFixed(2)} kWh`}
        />
        )}
        {totalSolarPowerGenerated !== null && (
        <MiniStatistics name='Solar Contribution' value={`${totalSolarPowerGenerated.toFixed(2)} kWh`} />     
        )}          
      </SimpleGrid>
      {pieChartData.length !== 0 && (
      <SimpleGrid columns={{ base: 1, md: 1, xl: 1 }} gap='20px' mb='20px'>
        <SimpleGrid columns={{ base: 1, md: 1, xl: 1 }} gap='20px'>
          <PieCard pieChartData={pieChartData}/>
        </SimpleGrid>
      </SimpleGrid>
    )}
    </Box>
  );
}
