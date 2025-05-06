package nl.uu.trafficmas.tests.roadNetwork;

import static org.junit.Assert.assertEquals;
import nl.uu.trafficmas.roadnetwork.Node;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;

import org.junit.Test;

public class RemoveNodeTest {

	@Test
	public void removeNode() {
		RoadNetwork rn 	= new RoadNetwork();
		Node n 			= new Node("test",0,0);
		rn.addNode(n);
		rn.removeNode(n);
		assertEquals(rn.getNodes().length, 0);
	}

}
