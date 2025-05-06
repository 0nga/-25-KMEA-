package nl.uu.trafficmas.tests.roadNetwork;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import nl.uu.trafficmas.roadnetwork.AStar;
import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.Lane;
import nl.uu.trafficmas.roadnetwork.LaneType;
import nl.uu.trafficmas.roadnetwork.Node;
import nl.uu.trafficmas.roadnetwork.Road;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;

import org.junit.Test;

public class AStarTest {

	
	@Test
	public void heuristicIdenticalTest() {
		Node from = new Node("from",5,5);
		double maxSpeed = 1;
		double travelTime = AStar.heuristic(from, from, maxSpeed);
		assertEquals(0,travelTime,0);
	}
	
	@Test
	public void heuristicOneDistDiffTest() {
		Node from = new Node("from",5,5);
		Node to = new Node("to",5,6);
		double maxSpeed = 1;
		double travelTime = AStar.heuristic(from, to, maxSpeed);
		assertEquals(1,travelTime,0);
	}

	@Test
	public void heuristicTwoSqrtDistDiffTest() {
		Node from = new Node("from",5,5);
		Node to = new Node("to",6,6);
		double maxSpeed = 1;
		double travelTime = AStar.heuristic(from, to, maxSpeed);
		assertEquals(Math.sqrt(2),travelTime,0);
	}
	
	@Test
	public void heuristicOneDistDiffTenSpeedTest() {
		Node from = new Node("from",5,5);
		Node to = new Node("to",6,6);
		double maxSpeed = 10;
		double travelTime = AStar.heuristic(from, to, maxSpeed);
		assertEquals(Math.sqrt(2)/maxSpeed,travelTime,0);
	}
	
	@Test
	public void findShortestPathSameNodesTest() {
		RoadNetwork rn = new RoadNetwork();
		Node from = new Node("from",5,5);
		rn.addNode(from);
		double maxSpeed = 1;
		Map<String, Double> averageTravelTime = null;
		List<String> path = AStar.findShortestPath(from, from, rn, averageTravelTime , maxSpeed);
		assertNull(path);
	}
	
	@Test
	public void findShortestPathOneApartNodesTest() {
		RoadNetwork rn = new RoadNetwork();
		Node from = new Node("from",5,5);
		Node to = new Node("to",5,6);
		Lane l = new Lane(LaneType.Normal, (byte) 0);
		List<Lane> laneList = new ArrayList<Lane>();
		laneList.add(l);
		Road r = new Road("from_to", Node.nodeDistance(from, to), laneList, 0);
		Edge edge = new Edge(from, to, r);
		rn.addNode(from);
		rn.addNode(to);
		rn.addEdge(edge);
		double maxSpeed = 1;
		Map<String, Double> averageTravelTime = new HashMap<String, Double>();
		averageTravelTime.put(edge.getID(), 100.0);// file
		List<String> path = AStar.findShortestPath(from, to, rn, averageTravelTime , maxSpeed);
		assertNotNull(path);
		assertEquals(1,path.size());
		assertEquals(edge.getID(),path.get(0));
	}
	
	@Test
	public void findShortestPathTwoOptionsOneJamNodesTest() {
		RoadNetwork rn 	= new RoadNetwork();
		Node from 		= new Node("from",5,5);
		Node to1 		= new Node("to1",5,6);
		Node to2 		= new Node("to2",6,6);
		Node dest   	= new Node("dest",5,7);
		
		rn.addNode(from);
		rn.addNode(to1);
		rn.addNode(to2);
		rn.addNode(dest);

		Edge fromTo1Edge 	= RoadNetwork.createSimpleEdge(rn, from, to1, "from_to1");
		Edge fromTo2Edge 	= RoadNetwork.createSimpleEdge(rn, from, to2, "from_to2");
		Edge to1ToDest 		= RoadNetwork.createSimpleEdge(rn, to1, dest, "1_todest");
		Edge to2ToDest 		= RoadNetwork.createSimpleEdge(rn, to2, dest, "2_todest");

		Map<String, Double> averageTravelTime = new HashMap<String, Double>();
		averageTravelTime.put(fromTo1Edge.getID(), 100.0);// file
		averageTravelTime.put(fromTo2Edge.getID(), 10.0);// no file
		averageTravelTime.put(to1ToDest.getID(), 100.0);// no
		averageTravelTime.put(to2ToDest.getID(), 10.0);// file

		double maxSpeed = 1;

		List<String> path = AStar.findShortestPath(from, dest, rn, averageTravelTime , maxSpeed);
		
		assertNotNull(path);
		// we expect that the fastest route is now to take the longer roads.
		assertEquals(2,path.size());
		assertEquals(fromTo2Edge.getID(),path.get(0));
		assertEquals(to2ToDest.getID(),path.get(1));
	}

	@Test
	public void findShortestPathTwoOptionsTwoJamsNodesTest() {
		RoadNetwork rn 	= new RoadNetwork();
		Node from 		= new Node("from",5,5);
		Node to1 		= new Node("to1",5,6);
		Node to2 		= new Node("to2",6,6);
		Node dest   	= new Node("dest",5,7);
		
		rn.addNode(from);
		rn.addNode(to1);
		rn.addNode(to2);
		rn.addNode(dest);

		Edge fromTo1Edge 	= RoadNetwork.createSimpleEdge(rn, from, to1, "from_to1");
		Edge fromTo2Edge 	= RoadNetwork.createSimpleEdge(rn, from, to2, "from_to2");
		Edge to1ToDest 		= RoadNetwork.createSimpleEdge(rn, to1, dest, "1_todest");
		Edge to2ToDest 		= RoadNetwork.createSimpleEdge(rn, to2, dest, "2_todest");

		Map<String, Double> averageTravelTime = new HashMap<String, Double>();
		averageTravelTime.put(fromTo1Edge.getID(), 100.0);// file
		averageTravelTime.put(fromTo2Edge.getID(), 10.0);// no file
		averageTravelTime.put(to1ToDest.getID(), 9.0);// no
		averageTravelTime.put(to2ToDest.getID(), 100.0);// file

		
		double maxSpeed = 1;

		List<String> path = AStar.findShortestPath(from, dest, rn, averageTravelTime , maxSpeed);
		
		assertNotNull(path);
		// we expect that the fastest route is now to take the shorter roads.
		assertEquals(2,path.size());
		assertEquals(fromTo1Edge.getID(),path.get(0));
		assertEquals(to1ToDest.getID(),path.get(1));
	}
	
}
