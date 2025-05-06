package nl.uu.trafficmas.tests.roadNetwork;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import nl.uu.trafficmas.roadnetwork.Node;

import org.junit.Test;

public class NodeEqual {

	@Test
	public void nodeEqual() {
		Node n1 = new Node("a",0,0);
		Node n2 = new Node("a", 0, 0);
		assertEquals(n1,n2);
		
		Node n3 = new Node("b",0,0);
		assertNotEquals(n1,n3);
		assertNotEquals(n2,n3);
		
		Node n4 = new Node("a",1,0);
		assertNotEquals(n1,n4);
		assertNotEquals(n2,n4);

		Node n5 = new Node("a",0,1);
		assertNotEquals(n1,n5);
		assertNotEquals(n2,n5);
	}

}
