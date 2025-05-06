package nl.uu.trafficmas.tests.agents;

import static org.junit.Assert.assertEquals;
import it.polito.appeal.traci.SumoTraciConnection;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
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

import de.tudresden.sumo.cmd.Vehicle;

public class DoActionTest {

	@Test
	public void doAction() throws SAXException, IOException, ParserConfigurationException {
		Random random = new Random(1337);

		String dir 			= System.getProperty("user.dir")+"/tests/Agent/DoAction/";
		DataModel dataModel = new DataModelXML(dir,"MASTest.xml");
		MASData masData 	= dataModel.getMASData(); 
		
		HashMap<String, String> options = new HashMap<String, String>();
		options.put("e", Integer.toString(masData.simulationLength));
		options.put("start", "1");
		options.put("quit-on-end", "1");
		
		SumoTraciConnection conn 	= SimulationModelTraaS.initializeWithOptions(options,"sumo", System.getProperty("user.dir")+"/tests/Agent/DoAction/ConfigTest.xml");
		RoadNetwork rn 				= dataModel.instantiateRoadNetwork();
		ArrayList<Route> routes 	= dataModel.getRoutes(rn);
		
		HashMap<Agent,Integer> agentPairList 	= TrafficMASController.instantiateAgents(masData, random, routes, rn);
		HashMap<String, Agent> completeAgentMap = SimulationModelTraaS.addAgents(agentPairList, random, -1, conn);
		HashMap<String, Agent> currentAgentMap 	= SimulationModelTraaS.updateCurrentAgentMap(completeAgentMap, new HashMap<String, Agent>(), conn);
		
		boolean timeStep = false;
		StateData stateData = SimulationModelTraaS.getStateData(conn, timeStep);

		// The first agent is a granny
		Agent granny = completeAgentMap.get("Agent 0");
		
		// The fourth is a HotShot
		Agent hotshot = completeAgentMap.get("Agent 3");
		
		
		int i = 0;
		while(i++ < masData.simulationLength){
			currentAgentMap = SimulationModelTraaS.updateCurrentAgentMap(completeAgentMap, currentAgentMap, conn);
			stateData 		= SimulationModelTraaS.getStateData(conn, true);
			currentAgentMap = TrafficMASController.updateAgents(completeAgentMap, rn, stateData);
			rn = TrafficMASController.updateRoadNetwork(rn, stateData);
			try {			
				if(currentAgentMap.containsValue(granny) && currentAgentMap.containsValue(hotshot)){
					AgentAction grannyAction 	= granny.doAction(stateData.currentTimeStep/1000,null,null,null);
					//AgentAction hotShotAction 	= hotshot.doAction(stateData.currentTimeStep/1000,null,null,null);
					assertEquals(AgentAction.ChangeVelocity20, grannyAction);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}	
		}
	}
}
