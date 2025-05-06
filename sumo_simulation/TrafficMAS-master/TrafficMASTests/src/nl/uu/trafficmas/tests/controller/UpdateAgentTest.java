package nl.uu.trafficmas.tests.controller;

import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import it.polito.appeal.traci.SumoTraciConnection;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.controller.TrafficMASController;
import nl.uu.trafficmas.datamodel.DataModel;
import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.datamodel.MASData;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Route;
import nl.uu.trafficmas.simulationmodel.SimulationModelTraaS;
import nl.uu.trafficmas.simulationmodel.StateData;

import org.junit.Test;
import org.xml.sax.SAXException;

public class UpdateAgentTest {

	@Test
	public void updateAgent() throws SAXException, IOException, ParserConfigurationException {
		Random random 	= new Random(1337);
		String dir		= System.getProperty("user.dir")+"/tests/Controller/UpdateAgent/";
		String sumocfg 	= System.getProperty("user.dir")+"/tests/Controller/UpdateAgent/ConfigTest.xml";
		String masXML 	= "MASTest.xml";
		
		DataModel dataModel = new DataModelXML(dir,masXML);
		MASData masData 	= dataModel.getMASData(); 
		
		HashMap<String, String> options = new LinkedHashMap<String, String>();
		options.put("e", Integer.toString(masData.simulationLength));
		options.put("start", "1");
		options.put("quit-on-end", "1");
		
		SumoTraciConnection conn = SimulationModelTraaS.initializeWithOptions(options,"sumo", sumocfg);
		
		RoadNetwork rn 			= dataModel.instantiateRoadNetwork();
		ArrayList<Route> routes = dataModel.getRoutes(rn);
		
		HashMap<Agent,Integer> agentPairList 	= TrafficMASController.instantiateAgents(masData, random, routes, rn);
		HashMap<String, Agent> completeAgentMap = SimulationModelTraaS.addAgents(agentPairList, random, -1, conn);
		StateData stateData 					= SimulationModelTraaS.getStateData(conn, false);
		HashMap<String, Agent> currentAgentMap 	= new HashMap<String, Agent>();
	
		int i = 0;
		while(i++ < masData.simulationLength){
			try {
				if(stateData.agentsData.size() == 2) {
					for(Map.Entry<String, Agent> entry : currentAgentMap.entrySet()){
						TrafficMASController.updateAgent(rn, stateData, entry.getValue().agentID, entry.getValue(), completeAgentMap);
						assertNotEquals(0,entry.getValue().getVelocity(),0);
						assertNotNull(entry.getValue().getRoad());
						assertNotNull(entry.getValue().getLane());		
						assertTrue(entry.getValue().getExpectedArrivalTime()*1000 > stateData.currentTimeStep/1000);
					}
				}
				stateData 		= SimulationModelTraaS.getStateData(conn, true);
				currentAgentMap = SimulationModelTraaS.updateCurrentAgentMap(completeAgentMap, currentAgentMap, conn);

			} catch (Exception e) {
				e.printStackTrace();
			}	
		}
	}
}
