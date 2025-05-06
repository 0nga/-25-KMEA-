package nl.uu.trafficmas.tests.roadNetwork;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import nl.uu.trafficmas.roadnetwork.Node;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;

import org.junit.Test;

public class AddNodesTest {

	@Test
	public void addNodes() {
		RoadNetwork rn = new RoadNetwork();
		
		Node n1 = new Node("n1",0,0);
		Node n2 = new Node("n2",0,0);
		
		ArrayList<Node> nodes = new ArrayList<Node>();
		nodes.add(n1);
		nodes.add(n2);
		
		rn.addNodes(nodes);
		
		Set<Node> nodeSet = new HashSet<Node>();
		for(Node e: rn.getNodes()) {
			nodeSet.add(e);
		}
		
		assertEquals(rn.getNodes().length, 2);
		assertEquals(true, nodeSet.contains(n1));
		assertEquals(true, nodeSet.contains(n2));
	}
}
