package nl.uu.trafficmas.tests.roadNetwork;

import static org.junit.Assert.assertNotNull;
import nl.uu.trafficmas.roadnetwork.Node;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;

import org.junit.Test;

public class GetEdgeTest {

	@Test
	public void test() {
		RoadNetwork rn 	= new RoadNetwork();
		Node from 		= new Node("from",5,5);
		Node to1 		= new Node("to1",5,6);
		Node to2 		= new Node("to2",6,6);
		Node dest   	= new Node("dest",5,7);
		
		rn.addNode(from);
		rn.addNode(to1);
		rn.addNode(to2);
		rn.addNode(dest);

		RoadNetwork.createSimpleEdge(rn, from, to1, "from_to1");
		RoadNetwork.createSimpleEdge(rn, from, to2, "from_to2");
		RoadNetwork.createSimpleEdge(rn, to1, dest, "1_todest");
		RoadNetwork.createSimpleEdge(rn, to2, dest, "2_todest");
		
		assertNotNull(rn.getEdge(from, to1));
		assertNotNull(rn.getEdge(from, to2));
		assertNotNull(rn.getEdge(to1, dest));
		assertNotNull(rn.getEdge(to2, dest));
		
	}

}
