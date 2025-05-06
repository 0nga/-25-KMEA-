package nl.uu.trafficmas.tests.roadNetwork;

import static org.junit.Assert.assertEquals;

import java.io.IOException;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.roadnetwork.Road;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;

import org.junit.Test;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class GetRoadFromIDTest {

	@Test
	public void getRoadFromID() throws ParserConfigurationException, SAXException, IOException {
		String dir = System.getProperty("user.dir")+"/tests/RoadNetwork/";
		Document nodeDoc = DataModelXML.loadDocument(dir, "NodeTest.xml");
		Document edgeDoc = DataModelXML.loadDocument(dir, "EdgeTest.xml");
		RoadNetwork rn = DataModelXML.instantiateRoadNetwork(nodeDoc, edgeDoc);
		Road r = rn.getRoadFromID("A28Tot350");
		assertEquals("A28Tot350", r.id);
		assertEquals(2, r.laneList.size());
	}
}
