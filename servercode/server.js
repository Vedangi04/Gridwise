const express = require('express');
const cors = require('cors');
const fs = require('fs');
const csv = require('csv-parser');
const ExcelJS = require('exceljs');

const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

const winddata = [];
const csvFilePath = 'test_predictions.csv';
const machinedata = [];
const csvFilePath1 = 'machine_test_data.csv';
const solardata = [];
const csvFilePath2 = 'test_solar_predictions.csv';

fs.createReadStream(csvFilePath)
  .pipe(csv())
  .on('data', (row) => {
    winddata.push(row);
  });

fs.createReadStream(csvFilePath1)
  .pipe(csv())
  .on('data', (row) => {
    machinedata.push(row);
  });

fs.createReadStream(csvFilePath2)
  .pipe(csv())
  .on('data', (row) => {
    solardata.push(row);
  });

app.post('/api/getprediction', async (req, res) => {
  try {
    const criteria = req.body.filterCriteria;
    console.log('CCCCCCC ', criteria);

    const filteredData = winddata.filter(item => item.time.startsWith(criteria));
    const windpowerdata = filteredData.map(item => item.ActualPower);

    const averageActualPower = calculateAverageActualPower(windpowerdata);
    const maximumActualPower = calculateMaximumActualPower(windpowerdata);
    const totalWindPowerGenerated = calculateTotalWindPowerGenerated(filteredData);

    console.log('PPP ', totalWindPowerGenerated);

    res.json({ averageActualPower, maximumActualPower,totalWindPowerGenerated, apidata: filteredData });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/getgeneration', async (req, res) => {
  try {
    const criteria = req.body.filterCriteria;
    console.log('DDDDDDDD ', criteria);

    const filteredData = solardata.filter(item => item.time.startsWith(criteria));
    const solarpowerdata = filteredData.map(item => item.ActualPower);

    const averageActualPower = calculateAverageActualPower(solarpowerdata);
    const maximumActualPower = calculateMaximumActualPower(solarpowerdata);
    const totalSolarPowerGenerated = calculateTotalSolarPowerGenerated(filteredData);

    res.json({ averageActualPower, maximumActualPower,totalSolarPowerGenerated, apidata: filteredData });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/getconsumption', async (req, res) => {
  try {
    const criteria = req.body.filterCriteria;
    console.log('CCCCCCC ', criteria);

    const filteredData = machinedata.filter(item => item.time.startsWith(criteria));
    const totalPowerConsumption = calculateTotalPowerConsumption(filteredData);
    const anomalies = checkAnomalies(filteredData);
    console.log('aaaaaaaaaaaaa:', anomalies);

    console.log('Total Power Consumption:', totalPowerConsumption);
    res.json({ totalPowerConsumption,anomalies, apidata: filteredData });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/getcombinedtotalpower', async (req, res) => {
  try {
    const criteria = req.body.filterCriteria;
    console.log('Filter Criteria:', criteria);

    const filteredWindData = winddata.filter(item => item.time.startsWith(criteria));
    const totalWindPowerGenerated = calculateTotalWindPowerGenerated(filteredWindData);
    const filteredSolarData = solardata.filter(item => item.time.startsWith(criteria));
    const totalSolarPowerGenerated = calculateTotalSolarPowerGenerated(filteredSolarData);
    const combinedTotalPower = (totalWindPowerGenerated + totalSolarPowerGenerated).toFixed(2);
    res.json({ combinedTotalPower, totalWindPowerGenerated, totalSolarPowerGenerated });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/generateReport', async (req, res) => {
  try {
    const criteria = req.body.filterCriteria;
    console.log('Generating report for:', criteria);

    const filteredWindData = winddata.filter(item => item.time.startsWith(criteria));
    const filteredSolarData = solardata.filter(item => item.time.startsWith(criteria));

    console.log('Filtered Wind Data:', filteredWindData);
    console.log('Filtered Solar Data:', filteredSolarData);

    const combinedData = {};
    filteredWindData.forEach(item => {
      const key = item.time;
      if (!combinedData[key]) {
        combinedData[key] = {};
      }
      combinedData[key]['wind'] = item['PredictedPower'];
    });

    filteredSolarData.forEach(item => {
      const key = item.time;
      if (!combinedData[key]) {
        combinedData[key] = {};
      }
      combinedData[key]['solar'] = item['PredictedPower'];
    });

    const workbook = new ExcelJS.Workbook();
    const worksheet = workbook.addWorksheet('PredictedValues');
    worksheet.addRow(['Date', 'Time', 'Wind Predicted Power (kW)', 'Solar Predicted Power (kW)']);

    Object.entries(combinedData).forEach(([time, data]) => {
      const [date, hour] = time.split(' ');
      worksheet.addRow([date, hour, data.wind || '', data.solar || '']);
    });

    res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    res.setHeader('Content-Disposition', 'attachment; filename="predicted_values.xlsx"');
    res.end(await workbook.xlsx.writeBuffer());
  } catch (error) {
    console.error('Error generating report:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

const calculateTotalPowerConsumption = (filteredData) => {
  let totalPowerConsumption = 0;
  filteredData.forEach(item => {
    Object.keys(item).forEach(key => {
      if (key.indexOf('Energy Consumed') !== -1) {
        totalPowerConsumption += parseFloat(item[key]);
      }
    });
  });
  console.log('Total Power:', totalPowerConsumption);
  return totalPowerConsumption;
};

const calculateAverageActualPower = (filteredData) => {
  const numericValues = filteredData.map(parseFloat)
  const totalActualPower = numericValues.reduce((sum, item) => sum + item, 0);
  const averageActualPower = totalActualPower / filteredData.length;
  console.log('Average Actual Power:', averageActualPower);
  return averageActualPower;
};

const calculateMaximumActualPower = (filteredData) => {
  const numericValues = filteredData.map(parseFloat);
  const maxActualPower = Math.max(...numericValues);
  console.log('Maximum Actual Power:', maxActualPower);
  return maxActualPower;
};

const calculateTotalWindPowerGenerated = (filteredData) => {
  let totalWindPowerGenerated = 0;
  filteredData.forEach(item => {
    Object.keys(item).forEach(key => {
      if (key.indexOf('ActualPower') !== -1) { 
        totalWindPowerGenerated += parseFloat(item[key]);
      }
    });
  });
  console.log('Total Wind Power Generated:', totalWindPowerGenerated);
  return totalWindPowerGenerated;
};

const calculateTotalSolarPowerGenerated = (filteredData) => {
  let totalSolarPowerGenerated = 0;
  filteredData.forEach(item => {
    Object.keys(item).forEach(key => {
      if (key.indexOf('ActualPower') !== -1) { 
        totalSolarPowerGenerated += parseFloat(item[key]);
      }
    });
  });
  console.log('Total Solar Power Generated:', totalSolarPowerGenerated);
  return totalSolarPowerGenerated;
};

const machineConsumptionThreshold = 9.3 ; 
const temperatureThreshold = 48.94; 

const checkAnomalies = (filteredData) => {
  let anomalies = {
    consumptionAnomaly: false,
    temperatureAnomaly: false
  };

  for (let index = 0; index < 5; index++) {
    const consumptionKey = `Machine_${index + 1} Energy Consumed (kWh)`;
    const temperatureKey = `Machine_${index + 1} Temperature (C)`;

    filteredData.forEach(item => {
      if (consumptionKey in item && parseFloat(item[consumptionKey]) > machineConsumptionThreshold) {
        anomalies.consumptionAnomaly = true;
      }
      if (temperatureKey in item && parseFloat(item[temperatureKey]) > temperatureThreshold) {
        anomalies.temperatureAnomaly = true;
      }
    });
  }

  console.log('Anomalies:', anomalies);

  const anyAnomalyDetected = anomalies.consumptionAnomaly || anomalies.temperatureAnomaly;

  return anyAnomalyDetected ? 'Yes' : 'No';

};

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
