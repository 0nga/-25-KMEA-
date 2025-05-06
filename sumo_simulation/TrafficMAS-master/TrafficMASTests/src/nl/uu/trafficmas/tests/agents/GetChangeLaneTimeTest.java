package nl.uu.trafficmas.tests.agents;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.agent.actions.ChangeLaneAction;
import nl.uu.trafficmas.controller.TrafficMASController;
import nl.uu.trafficmas.datamodel.DataModel;
import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.datamodel.MASData;
import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.Lane;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Route;

import org.junit.Test;
import org.xml.sax.SAXException;

public class GetChangeLaneTimeTest {

	@Test
	public void getChangeLaneTime() throws SAXException, IOException, ParserConfigurationException {
		/*
		 * change lane time (no knowledge about RN)
		 * 
		 * lane_remainder       rest_route_length
		 * --------------   +   ----------------- + current_time
		 * next_lane_speed      max_comfy_speed
		 */
		// change lane time:
		Random random = new Random(1337);

		String dir 			= System.getProperty("user.dir")+"/tests/Agent/GetChangeLaneTime/";
		DataModel dataModel = new DataModelXML(dir,"MASTest.xml");
		MASData masData 	= dataModel.getMASData(); 
		
		RoadNetwork rn 				= dataModel.instantiateRoadNetwork();
		
		for(Edge e : rn.getEdges()) {
			e.getRoad().setMeanTravelTime(0.5);
			for(Lane lane : e.getRoad().getLanes()) {
				lane.setMeanTravelTime(0.5);
			}
		}
		ArrayList<Route> routes 	= dataModel.getRoutes(rn);
		
		HashMap<Agent,Integer> agentPairList 	= TrafficMASController.instantiateAgents(masData, random, routes, rn);
		ChangeLaneAction action = new ChangeLaneAction(0);
		for(Agent agent : agentPairList.keySet()) {
			agent.setRoad(rn.getEdges()[0].getRoad());
			assertEquals(5.5,action.getTime(5, 0, 2, 1, 2, 2, 1, 0, 0,agent),0);
		}
	}
}