package nl.uu.trafficmas.tests.misc;

import it.polito.appeal.traci.SumoTraciConnection;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Random;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.agent.AgentProfileType;
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

public class MergingTest {

	@Test
	public void test() throws SAXException, IOException, ParserConfigurationException {
		Random random 	= new Random(1337);
		String dir 		= System.getProperty("user.dir")+"/tests/Misc/MergingTest/";
		String sumocfg 	= System.getProperty("user.dir")+"/tests/Misc/MergingTest/hello.sumocfg";
		String masXML 	= "hello.mas.xml";
		
		DataModel dataModel = new DataModelXML(dir,masXML);
		MASData masData		= dataModel.getMASData();
		
		HashMap<String, String> options = new LinkedHashMap<String, String>();
		options.put("e", Integer.toString(masData.simulationLength));
		options.put("start", "1");
		options.put("quit-on-end", "1");
		
		SumoTraciConnection conn = SimulationModelTraaS.initializeWithOptions(options,"sumo", sumocfg);				
		RoadNetwork rn = dataModel.instantiateRoadNetwork();
		ArrayList<Route> routes = dataModel.getRoutes(rn);
		LinkedHashMap<AgentProfileType, Double> agentProfileDistribution = new LinkedHashMap<AgentProfileType, Double>();
		agentProfileDistribution.put(AgentProfileType.Normal, 1.0);
		 
		LinkedHashMap<Agent,Integer> agentsAndTimes 	= new LinkedHashMap<Agent, Integer>();
		// Route0 is main road
		TrafficMASController.createAgent(routes.get(0), agentsAndTimes, agentProfileDistribution, rn, 1, .5);
		TrafficMASController.createAgent(routes.get(0), agentsAndTimes, agentProfileDistribution, rn, 3, .5);
		TrafficMASController.createAgent(routes.get(0), agentsAndTimes, agentProfileDistribution, rn, 8, .5);
		
		
		// Route1 is the ramp road
		TrafficMASController.createAgent(routes.get(1), agentsAndTimes, agentProfileDistribution, rn, 10, .5);
		TrafficMASController.createAgent(routes.get(1), agentsAndTimes, agentProfileDistribution, rn, 12, .5);
//		TrafficMASController.createAgent(routes.get(1), agentsAndTimes, agentProfileDistribution, rn, 10, .5);
//		TrafficMASController.createAgent(routes.get(1), agentsAndTimes, agentProfileDistribution, rn, 15, .5);
//		TrafficMASController.createAgent(routes.get(1), agentsAndTimes, agentProfileDistribution, rn, 20, .5);
	
		
		HashMap<String, Agent> completeAgentMap = SimulationModelTraaS.addAgents(agentsAndTimes, random, -1, conn);
		HashMap<String, Agent> currentAgentMap 	= SimulationModelTraaS.updateCurrentAgentMap(completeAgentMap, new LinkedHashMap<String, Agent>(), conn);
		StateData stateData;
		HashMap<Agent, AgentAction> actions;
		
		try {
			int i = 0;
			while (i++ < masData.simulationLength) {
				currentAgentMap = SimulationModelTraaS.updateCurrentAgentMap(completeAgentMap, currentAgentMap, conn);
				stateData 		= SimulationModelTraaS.getStateData(conn, true);
				currentAgentMap = TrafficMASController.updateAgents(completeAgentMap, rn, stateData);
				rn 				= TrafficMASController.updateRoadNetwork(rn, stateData);
				actions 		= TrafficMASController.getAgentActions(stateData.currentTimeStep/1000, currentAgentMap,null,null,null);
				SimulationModelTraaS.simulateAgentActions(actions, conn);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
