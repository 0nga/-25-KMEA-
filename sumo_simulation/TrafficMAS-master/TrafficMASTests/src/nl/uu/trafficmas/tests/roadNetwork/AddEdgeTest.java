package nl.uu.trafficmas.tests.roadNetwork;

import static org.junit.Assert.assertEquals;

import java.util.HashSet;
import java.util.Set;

import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.Node;
import nl.uu.trafficmas.roadnetwork.Road;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;

import org.junit.Test;

public class AddEdgeTest {

	@Test
	public void addEdge() {
		RoadNetwork rn = new RoadNetwork();
		
		Node node = new Node("test",5,5);
		Road road = new Road("r",5.0,null,1);
		rn.addNode(node);
		Edge edge = new Edge(node,node,road);
		rn.addEdge(edge);
		assertEquals(1,rn.getEdges().length);
		assertEquals(rn.getEdges()[0], edge);
		
		Node node2 = new Node("test2",5,5);
		rn.addEdge(node, node2, road);	
		
		Set<Edge> edgeSet = new HashSet<Edge>();
		for(Edge e: rn.getEdges()) {
			edgeSet.add(e);
		}
		
		assertEquals(rn.getEdges().length, 2);
		
		assertEquals(true, edgeSet.contains(edge));
		assertEquals(true, edgeSet.contains(new Edge(node, node2, road)));
	}

}
