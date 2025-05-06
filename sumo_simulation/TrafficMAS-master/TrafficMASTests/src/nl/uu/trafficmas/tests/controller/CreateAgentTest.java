package nl.uu.trafficmas.tests.controller;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.agent.AgentProfileType;
import nl.uu.trafficmas.controller.TrafficMASController;
import nl.uu.trafficmas.datamodel.DataModel;
import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.datamodel.MASData;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Route;

import org.junit.Test;
import org.xml.sax.SAXException;


public class CreateAgentTest {

	@Test
	public void createAgent() throws SAXException, IOException, ParserConfigurationException {
		String dir = System.getProperty("user.dir")+"/tests/Controller/CreateAgent/";
		
		String masXML 		= "MASTest.xml";
		DataModel dataModel = new DataModelXML(dir,masXML);
		MASData masData 	= dataModel.getMASData();
		RoadNetwork rn 		= dataModel.instantiateRoadNetwork();
		
		ArrayList<Route> routes 						= dataModel.getRoutes(rn);
		LinkedHashMap<Agent, Integer> agentsAndTimes 	=  new LinkedHashMap<Agent, Integer>();
		HashMap<String,LinkedHashMap<AgentProfileType, Double>> routeAgentTypeSpawnDist = masData.routeAgentTypeSpawnDist;
		assertEquals(0, agentsAndTimes.size());
		
		TrafficMASController.createAgent(routes.get(0), agentsAndTimes, routeAgentTypeSpawnDist.get(routes.get(0).routeID), rn, 2, 0.5);
		
		assertEquals(1, agentsAndTimes.size());
		Agent a;
		
		// There only is one agent.
		for(Map.Entry<Agent,Integer> entry : agentsAndTimes.entrySet()){
			a = entry.getKey();			
			assertTrue(a.agentID.startsWith("Agent "));
			assertArrayEquals(routes.get(0).getRoute(),a.getRoute());
			assertNotNull(a.getMaxSpeed());
			assertTrue(Arrays.asList(rn.getNodes()).contains(a.getGoalNode()));
		}
	}

}
