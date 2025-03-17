import React from "react";

// Chakra imports
import { Flex, Image } from "@chakra-ui/react";

// Custom components
import gridwise from "assets/img/gridwise.png";
import { HSeparator } from "components/separator/Separator";

export function SidebarBrand() {
  return (
    <Flex align='center' direction='column'>
      <Image src={gridwise} w='200px' h='80px' />
      <HSeparator mb='20px' />
    </Flex>
  );
}

export default SidebarBrand;
