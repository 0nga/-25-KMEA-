package nl.uu.trafficmas.tests.controller;

import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Random;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.controller.TrafficMASController;
import nl.uu.trafficmas.datamodel.DataModel;
import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.datamodel.MASData;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Route;

import org.junit.Test;
import org.xml.sax.SAXException;

public class InstantiateAgentsTest {

	//TODO: Make test for new instantiateAgent functionality (Spawning agents on every route.)

	@Test
	public void instantiateAgents() throws SAXException, IOException, ParserConfigurationException {
		Random random = new Random(1337);
	
		String dir 		= System.getProperty("user.dir")+"/tests/Controller/InstantiateAgents/";
		String masXML 	= "MASTest.xml";
		
		DataModel dataModel = new DataModelXML(dir,masXML);
		MASData masData 	= dataModel.getMASData();
		
		RoadNetwork rn 			= dataModel.instantiateRoadNetwork();
		ArrayList<Route> routes = dataModel.getRoutes(rn);

		HashMap<Agent,Integer> agentPairList = TrafficMASController.instantiateAgents(masData, random, routes, rn);
		
		assertNotEquals(agentPairList.size(), 0);
		
		for(Entry<Agent, Integer> pair : agentPairList.entrySet()) {
			assertNotNull(pair.getKey());
			assertNotEquals(0,(int)pair.getValue());
		}
		
	}
}
