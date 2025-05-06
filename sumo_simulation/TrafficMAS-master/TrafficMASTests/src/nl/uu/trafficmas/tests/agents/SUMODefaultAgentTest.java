package nl.uu.trafficmas.tests.agents;

import static org.junit.Assert.*;
import it.polito.appeal.traci.SumoTraciConnection;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.agent.actions.AgentAction;
import nl.uu.trafficmas.controller.TrafficMASController;
import nl.uu.trafficmas.datamodel.DataModel;
import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.datamodel.MASData;
import nl.uu.trafficmas.organisation.Organisation;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Route;
import nl.uu.trafficmas.simulationmodel.SimulationModelTraaS;
import nl.uu.trafficmas.simulationmodel.StateData;

import org.junit.Test;
import org.xml.sax.SAXException;

public class SUMODefaultAgentTest {

	@Test
	public void SUMODefaultAgent() throws SAXException, IOException, ParserConfigurationException {
		Random random 	= new Random(1337);
		String dir 		= System.getProperty("user.dir")+"/tests/Agent/SUMODefaultAgent/";
		String sumocfg 	= System.getProperty("user.dir")+"/tests/Agent/SUMODefaultAgent/orgnormtest.cfg.xml";
		String masXML 	= "orgnormtest.mas.xml";
		
		DataModel dataModel = new DataModelXML(dir,masXML);
		MASData masData		= dataModel.getMASData();
		
		HashMap<String, String> options = new LinkedHashMap<String, String>();
		options.put("e", Integer.toString(masData.simulationLength));
		options.put("start", "1");
		options.put("quit-on-end", "1");
		
		SumoTraciConnection conn = SimulationModelTraaS.initializeWithOptions(options,"sumo", sumocfg);				
		RoadNetwork rn = dataModel.instantiateRoadNetwork();
		ArrayList<Route> routes = dataModel.getRoutes(rn);

		HashMap<Agent,Integer> agentPairList 	= TrafficMASController.instantiateAgents(masData, random, routes, rn);
		HashMap<String, Agent> completeAgentMap = SimulationModelTraaS.addAgents(agentPairList, random, -1, conn);
		HashMap<String, Agent> currentAgentMap 	= SimulationModelTraaS.updateCurrentAgentMap(completeAgentMap, new LinkedHashMap<String, Agent>(), conn);

		try {
			int i = 0;
			while (i++ < masData.simulationLength) {
				currentAgentMap 		= SimulationModelTraaS.updateCurrentAgentMap(completeAgentMap, currentAgentMap, conn);
				StateData stateData 	= SimulationModelTraaS.getStateData(conn, true);
				currentAgentMap 		= TrafficMASController.updateAgents(completeAgentMap, rn, stateData);
				rn						= TrafficMASController.updateRoadNetwork(rn, stateData);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
