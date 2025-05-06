package nl.uu.trafficmas.tests.controller;

import static org.junit.Assert.assertEquals;
import it.polito.appeal.traci.SumoTraciConnection;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.agent.actions.AgentAction;
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

public class GetAgentActionsTest {

	@Test
	public void test() throws SAXException, IOException, ParserConfigurationException {
		Random random = new Random(1337);

		String dir = System.getProperty("user.dir")+"/tests/Controller/GetAgentActions/";
		String masXML = "MASTest.xml";
		DataModel dataModel = new DataModelXML(dir,masXML);
		MASData masData = dataModel.getMASData(); 
		
		HashMap<String, String> options = new HashMap<String, String>();
		options.put("e", Integer.toString(masData.simulationLength));
		options.put("start", "1");
		options.put("quit-on-end", "1");
		
		SumoTraciConnection conn 	= SimulationModelTraaS.initializeWithOptions(options,"sumo", System.getProperty("user.dir")+"/tests/Controller/GetAgentActions/ConfigTest.xml");				
		RoadNetwork rn 				= dataModel.instantiateRoadNetwork();
		ArrayList<Route> routes 	= dataModel.getRoutes(rn); 
		
		HashMap<Agent,Integer> agentPairList	= TrafficMASController.instantiateAgents(masData, random, routes, rn);
		HashMap<String, Agent> completeAgentMap = SimulationModelTraaS.addAgents(agentPairList, random, -1, conn);
		HashMap<String, Agent> currentAgentMap 	= SimulationModelTraaS.updateCurrentAgentMap(completeAgentMap, new HashMap<String, Agent>(), conn);
		HashMap<Agent, AgentAction> agentActionMap;

		StateData stateData = SimulationModelTraaS.getStateData(conn, false);
		
		int i = 0;
		while(i++ < masData.simulationLength){
			currentAgentMap = SimulationModelTraaS.updateCurrentAgentMap(completeAgentMap, currentAgentMap, conn);
			stateData 		= SimulationModelTraaS.getStateData(conn, true);

			currentAgentMap = TrafficMASController.updateAgents(completeAgentMap, rn, stateData);
			rn 				= TrafficMASController.updateRoadNetwork(rn, stateData);
			agentActionMap 	= TrafficMASController.getAgentActions(stateData.currentTimeStep/1000, currentAgentMap,null,null,null);
			
			try {
				// The fourth agent will ChangeLane, the rest does not need to take action.
				if(stateData.currentTimeStep == 6000){
					for(Map.Entry<Agent, AgentAction> entry : agentActionMap.entrySet()){
						if(entry.getKey().getClass().getSimpleName().equals("OldLadyAgent")){
							assertEquals(AgentAction.ChangeVelocity20,entry.getValue());
						}
					}
				}
			} catch (Exception e) {
				e.printStackTrace();
			}	
		}
	}
}