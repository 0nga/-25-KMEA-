package nl.uu.trafficmas.tests.roadNetwork;

import static org.junit.Assert.assertEquals;

import java.util.Set;

import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.Node;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;

import org.junit.Test;

public class GetSuccessorsTest {

	@Test
	public void getSuccessorsTest() {
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
		
		Set<Node> successors = rn.getSuccessors(from);
		assertEquals(true, successors.contains(to1));
		assertEquals(true, successors.contains(to2));
	}

}
