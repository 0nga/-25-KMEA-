package nl.uu.trafficmas.tests.roadNetwork;

import static org.junit.Assert.assertEquals;
import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.Node;
import nl.uu.trafficmas.roadnetwork.Road;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;

import org.junit.Test;

public class ClearTest {

	@Test
	public void clear() {
		RoadNetwork rn = new RoadNetwork();
		
		Node n1 = new Node("n1",0,0);
		Node n2 = new Node("n2",0,0);
		Road r1 = new Road("r1",7.0,null,1);
		Edge e1 = new Edge(n1,n2,r1);
		
		rn.addNode(n1);
		rn.addNode(n2);
		rn.addEdge(e1);
		rn.clear();
		assertEquals(0, rn.getNodes().length);
		assertEquals(0, rn.getEdges().length);
	}

}
