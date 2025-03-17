import React from "react";

import { Icon } from "@chakra-ui/react";
import {
  MdBarChart,
  MdPerson,
  MdHome,
} from "react-icons/md";

import MainDashboard from "views/admin/default";
import WindPowerPrediction from "views/admin/windpowerprediction";
import MachineConsumption from "views/admin/machineconsumption";
import SolarPowerPrediction from "views/admin/solarpowerprediction";

const routes = [
  {
    name: "Main Dashboard",
    layout: "/admin",
    path: "/default",
    icon: <Icon as={MdHome} width='20px' height='20px' color='inherit' />,
    component: MainDashboard,
  },
  {
    name: "Wind Power",
    layout: "/admin",
    path: "/nft-WindPowerPrediction",
    icon: 
      <Icon
        as={MdBarChart}
        width='20px'
        height='20px'
        color='inherit'
      />,
    component: WindPowerPrediction,
  },
  {
    name: "Solar Power",
    layout: "/admin",
    icon: <Icon as={MdBarChart} width='20px' height='20px' color='inherit' />,
    path: "/data-tables",
    component: SolarPowerPrediction,
  },
  {
    name: "Machine Consumption",
    layout: "/admin",
    path: "/profile",
    icon: <Icon as={MdPerson} width='20px' height='20px' color='inherit' />,
    component: MachineConsumption,
  },
];

export default routes;
