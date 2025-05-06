package nl.uu.trafficmas.tests.XML;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.HashMap;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Sensor;

import org.junit.Test;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class GetSensorsTest {

	@Test
	public void test() throws ParserConfigurationException, SAXException, IOException {
		String dir 			= System.getProperty("user.dir")+"/tests/XML/GetSensors/";
		String nodeXML 		= "NodeTest.xml";
		String edgeXML 		= "EdgeTest.xml";
		String sensorXML 	= "SensorTest.xml";
		
		Document nodeDoc 	= DataModelXML.loadDocument(dir, nodeXML);
		Document edgeDoc 	= DataModelXML.loadDocument(dir, edgeXML);
		Document sensorDoc	= DataModelXML.loadDocument(dir, sensorXML);

		RoadNetwork rn = DataModelXML.instantiateRoadNetwork(nodeDoc, edgeDoc);
		
		HashMap<String,Sensor> sensors = DataModelXML.getSensors(rn, sensorDoc);
		
		assertEquals(sensors.size(),1);
		assertEquals("A28_350_lane0_0",								sensors.get("A28_350_lane0_0").getId());
		assertEquals(10,											sensors.get("A28_350_lane0_0").getLength(),0.0);
		assertEquals(25, 											sensors.get("A28_350_lane0_0").getFrequency());
		assertEquals(300,											sensors.get("A28_350_lane0_0").getPosition(),0.0);
		assertEquals(rn.getRoadFromID("A28Tot350").laneList.get(0), sensors.get("A28_350_lane0_0").getLane());
	}
}
