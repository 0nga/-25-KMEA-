package nl.uu.trafficmas.tests.controller;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import it.polito.appeal.traci.SumoTraciConnection;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
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
import nl.uu.trafficmas.simulationmodel.AgentData;
import nl.uu.trafficmas.simulationmodel.SimulationModelTraaS;
import nl.uu.trafficmas.simulationmodel.StateData;

import org.junit.Test;
import org.xml.sax.SAXException;

public class UpdateAgentsTest {

	@Test
	public void updateAgents() throws SAXException, IOException, ParserConfigurationException {
		Random random 	= new Random(1337);
		String dir 		= System.getProperty("user.dir")+"/tests/Controller/UpdateAgents/";
		String masXML 	= "MASTest.xml";
		
		DataModel dataModel = new DataModelXML(dir,masXML);
		MASData masData 	= dataModel.getMASData(); 
		
		HashMap<String, String> options = new LinkedHashMap<String, String>();
		options.put("e", Integer.toString(masData.simulationLength));
		options.put("start", "1");
		options.put("quit-on-end", "1");
		
		String sumocfg = System.getProperty("user.dir")+"/tests/Controller/UpdateAgent/ConfigTest.xml";
		SumoTraciConnection conn = SimulationModelTraaS.initializeWithOptions(options,"sumo", sumocfg);				
		
		RoadNetwork rn 			= dataModel.instantiateRoadNetwork();
		ArrayList<Route> routes = dataModel.getRoutes(rn);
		
		HashMap<Agent,Integer> agentPairList 	= TrafficMASController.instantiateAgents(masData, random, routes, rn);
		HashMap<String, Agent> completeAgentMap = SimulationModelTraaS.addAgents(agentPairList, random, -1, conn);
		boolean timeStep 						= false;
		StateData stateData 					= SimulationModelTraaS.getStateData(conn, timeStep);
		HashMap<String, Agent>updatedAgentMap 	= TrafficMASController.updateAgents(completeAgentMap, rn, stateData);

		// All agents have a velocity of 0.
		for(Entry<String, Agent> val : completeAgentMap.entrySet()) {
			assertEquals(0.0,val.getValue().getVelocity(),0);
		}

		int i = 0;
		while(i++ < masData.simulationLength){
			try {
				stateData = SimulationModelTraaS.getStateData(conn, true);
				if(stateData.agentsData.size() > 2) {
					for(Entry<String, AgentData> val : stateData.agentsData.entrySet()) {
						updatedAgentMap = TrafficMASController.updateAgents(completeAgentMap, rn, stateData);
						assertNotEquals(0.0,updatedAgentMap.get(val.getKey()).getVelocity());
					}
					break;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}	
		}
		
	
	}
}
