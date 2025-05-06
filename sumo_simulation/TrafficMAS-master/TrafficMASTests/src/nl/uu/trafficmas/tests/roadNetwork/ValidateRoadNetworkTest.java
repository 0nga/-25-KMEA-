package nl.uu.trafficmas.tests.roadNetwork;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;

import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.Node;
import nl.uu.trafficmas.roadnetwork.Road;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;

import org.junit.Test;

public class ValidateRoadNetworkTest {

	@Test
	public void validateRoadNetwork() {
		RoadNetwork rn = new RoadNetwork();
		Node n1 = new Node("n1",0,0);
		Node n2 = new Node("n2",0,0);
		Node n3 = new Node("n3",0,0);
		ArrayList<Node> nodes = new ArrayList<Node>();
		nodes.add(n1);
		nodes.add(n2);
		nodes.add(n3);
		Road r1 = new Road("r1", 2.0,null,1);
		Road r2 = new Road("r2", 5.0,null,1);
		Edge e1 = new Edge(n1,n1,r1);
		Edge e2 = new Edge(n1,n2,r1);
		Edge e3 = new Edge(n2,n3,r1);
		Edge e4 = new Edge(n2,n3,r2);
		
		// No duplicate nodes
		rn.addNode(n1);
		rn.addNode(n1);
		assertTrue(rn.validateRoadNetwork());
		assertEquals(1,rn.getNodes().length);
		rn.removeNode(n1);
		assertTrue(rn.validateRoadNetwork());
		rn.clear();
		
		// No duplicate edges
		rn.addNode(n1);
		rn.addNode(n2);
		rn.addEdge(e2);
		rn.addEdge(e2);
		assertTrue(rn.validateRoadNetwork());
		rn.removeEdge(e2);
		assertTrue(rn.validateRoadNetwork());
		rn.clear();
		
		// No edge from and to the same node
		rn.addNode(n1);
		rn.addEdge(e1);
		assertFalse(rn.validateRoadNetwork());
		rn.removeEdge(e1);
		assertTrue(rn.validateRoadNetwork());
		rn.clear();
		
		// An edge goes from and to nodes that are present in the RoadNetwork
		rn.addEdge(e2);
		assertTrue(rn.validateRoadNetwork());
		rn.addNode(n1);
		rn.addNode(n2);
		assertTrue(rn.validateRoadNetwork());
		rn.clear();
		
		// Every edge must have an unique road
		rn.addNodes(nodes);
		rn.addEdge(e2);
		rn.addEdge(e3);
		assertFalse(rn.validateRoadNetwork());
		rn.removeEdge(e3);
		rn.addEdge(e4);
		assertTrue(rn.validateRoadNetwork());
		rn.clear();
		
		//TODO Every road must have an unique ID
	}
}
