package nl.uu.trafficmas.tests.XML;

import static org.junit.Assert.assertEquals;

import java.io.IOException;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Sensor;

import org.junit.Test;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class XMLRoadNetworkTest {

	@Test
	public void xmlRoadNetwork() throws ParserConfigurationException, SAXException, IOException {
		String dir = System.getProperty("user.dir")+"/tests/XML/InstantiateRoadNetwork/";
		String nodeXML 		= "NodeTest.xml";
		String edgeXML 		= "EdgeTest.xml";
		
		Document nodeDoc 	= DataModelXML.loadDocument(dir, nodeXML);
		Document edgeDoc 	= DataModelXML.loadDocument(dir, edgeXML);
		RoadNetwork rn = DataModelXML.instantiateRoadNetwork(nodeDoc, edgeDoc);
		
		assertEquals(rn.getNodes().length,2);
		assertEquals(rn.getEdges().length,1);
		
		String sensorXML 	= "SensorTest.xml";
		
		Document sensorDoc 	= DataModelXML.loadDocument(dir, sensorXML);
		
		rn = DataModelXML.instantiateRoadNetwork(nodeDoc, edgeDoc, sensorDoc);
		
		Sensor sensor = rn.getSensorMap().get("A28_350_lane0_0");
		
		assertEquals("A28_350_lane0_0",								sensor.getId());
		assertEquals(10,											sensor.getLength(),0.0);
		assertEquals(25, 											sensor.getFrequency());
		assertEquals(300,											sensor.getPosition(),0.0);
		assertEquals(rn.getRoadFromID("A28Tot350").laneList.get(0), sensor.getLane());
	}
}
