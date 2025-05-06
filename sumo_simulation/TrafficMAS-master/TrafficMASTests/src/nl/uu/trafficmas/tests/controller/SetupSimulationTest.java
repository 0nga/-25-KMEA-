package nl.uu.trafficmas.tests.controller;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.controller.TrafficMASController;
import nl.uu.trafficmas.datamodel.DataModel;
import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.datamodel.MASData;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Route;
import nl.uu.trafficmas.simulationmodel.SimulationModel;
import nl.uu.trafficmas.simulationmodel.SimulationModelTraaS;

import org.junit.Test;
import org.xml.sax.SAXException;

public class SetupSimulationTest {

	@Test
	public void setupSimulation() throws SAXException, IOException, ParserConfigurationException {
		Random rng 		= new Random(1337);
		String dir 		= System.getProperty("user.dir")+"/tests/Controller/SetupSimulation/";
		String masXML 	= "MASTest.xml";
		String sumocfg 	= System.getProperty("user.dir")+"/tests/Controller/SetupSimulation/ConfigTest.xml";

		DataModel dataModel		 = new DataModelXML(dir,masXML);
		SimulationModel simModel = new SimulationModelTraaS("sumo",sumocfg);
		
		MASData masData 		= dataModel.getMASData();
		RoadNetwork rn 			= dataModel.instantiateRoadNetwork();
		ArrayList<Route> routes = dataModel.getRoutes(rn);

		HashMap<Agent,Integer> agentsAndTime 	= TrafficMASController.instantiateAgents(masData, rng, routes, rn);
		HashMap<String, Agent> completeAgentMap = TrafficMASController.setupSimulation(masData, simModel, agentsAndTime, rng);
		
		assertEquals(9, completeAgentMap.size());

	}

}
