package nl.uu.trafficmas.tests.simulationModelTraaS;

import static org.junit.Assert.assertNotNull;
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

public class QueryBuilderTest {

	@Test
	public void queryBuilder() throws SAXException, IOException, ParserConfigurationException {
		Random random 	= new Random(1337);
		String dir 		= System.getProperty("user.dir")+"/tests/SimulationModelTraaS/QueryBuilder/";
		String masXML 	= "MASTest.xml";
		String sumocfg 	= System.getProperty("user.dir")+"/tests/SimulationModelTraaS/QueryBuilder/ConfigTest.xml";

		DataModel dataModel = new DataModelXML(dir,masXML);
		MASData masData 	= dataModel.getMASData(); 
		
		HashMap<String, String> options = new LinkedHashMap<String, String>();
		options.put("e", Integer.toString(masData.simulationLength));
		options.put("start", "1");
		options.put("quit-on-end", "1");
		
		SumoTraciConnection conn = SimulationModelTraaS.initializeWithOptions(options,"sumo", sumocfg);
		
		RoadNetwork rn 			= dataModel.instantiateRoadNetwork();
		ArrayList<Route> routes = dataModel.getRoutes(rn);
		
		HashMap<Agent,Integer> agentPairList = TrafficMASController.instantiateAgents(masData, random, routes, rn);
		HashMap<String, Agent> completeAgentMap = SimulationModelTraaS.addAgents(agentPairList, random, -1, conn);
		
		boolean timeStep 	= false;
		StateData stateData = SimulationModelTraaS.getStateData(conn, timeStep);
		assertNotNull(stateData);
		int i = 0;
		while(i++ < masData.simulationLength){
			timeStep = true;
			stateData = SimulationModelTraaS.getStateData(conn, timeStep);
			if(!stateData.agentsData.isEmpty()) {
				for(Entry<String, AgentData> val : stateData.agentsData.entrySet()) {
					assertNotNull(val.getValue().laneIndex);
					assertNotNull(val.getValue().position);
					assertNotNull(val.getValue().velocity);
				}
			}
		}
	}

}
