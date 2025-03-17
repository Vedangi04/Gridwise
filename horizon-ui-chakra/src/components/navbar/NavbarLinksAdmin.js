// Chakra Imports
import {
	Flex,
	Icon,
	Menu,
	MenuButton,
	MenuList,
	Text,
	useColorModeValue
} from '@chakra-ui/react';
// Custom Components
import { SearchBar } from 'components/navbar/searchBar/SearchBar';
import { SidebarResponsive } from 'components/sidebar/Sidebar';
import PropTypes from 'prop-types';
import React from 'react';
// Assets
import { MdNotificationsNone } from 'react-icons/md';
import routes from 'routes.js';
export default function HeaderLinks(props) {
	const { secondary, anomalyPresent } = props; // Add anomalyPresent prop
  
	// Chakra Color Mode
	const navbarIcon = useColorModeValue('gray.400', 'white');
	let menuBg = useColorModeValue('white', 'navy.800');
	const textColor = useColorModeValue('secondaryGray.900', 'white');
	const shadow = useColorModeValue(
	  '14px 17px 40px 4px rgba(112, 144, 176, 0.18)',
	  '14px 17px 40px 4px rgba(112, 144, 176, 0.06)'
	);
  
	return (
	  <Flex
		w={{ sm: '100%', md: 'auto' }}
		alignItems="center"
		flexDirection="row"
		bg={menuBg}
		flexWrap={secondary ? { base: 'wrap', md: 'nowrap' } : 'unset'}
		p="10px"
		borderRadius="30px"
		boxShadow={shadow}
	  >
		<SearchBar mb={secondary ? { base: '10px', md: 'unset' } : 'unset'} me="10px" borderRadius="30px" />
		<SidebarResponsive routes={routes} />
		<Menu>
		  <MenuButton p="0px">
			<Icon mt="6px" as={MdNotificationsNone} color={navbarIcon} w="18px" h="18px" me="10px" />
		  </MenuButton>
		  <MenuList
			boxShadow={shadow}
			p="20px"
			borderRadius="20px"
			bg={menuBg}
			border="none"
			mt="22px"
			me={{ base: '30px', md: 'unset' }}
			minW={{ base: 'unset', md: '400px', xl: '450px' }}
			maxW={{ base: '360px', md: 'unset' }}
		  >
			<Flex justify="space-between" w="100%" mb="20px">
			  <Text fontSize="md" fontWeight="600" color={textColor}>
				{anomalyPresent ? 'Anomaly Present' : 'Notifications'}
			  </Text>
			</Flex>
		  </MenuList>
		</Menu>
	  </Flex>
	);
  }
  
  HeaderLinks.propTypes = {
	variant: PropTypes.string,
	fixed: PropTypes.bool,
	secondary: PropTypes.bool,
	onOpen: PropTypes.func,
	anomalyPresent: PropTypes.bool, // Add anomalyPresent prop type
  };
  
