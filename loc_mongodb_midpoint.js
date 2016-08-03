
// Converts from degrees to radians. Needed with lat long calcs. 
Math.toRadians = function(degrees) {
  return degrees * Math.PI / 180;
}

// method adapted from http://www.movable-type.co.uk/scripts/latlong.html 
function haversine(lat1,lng1,lat2,lng2) {

    var R = 6371e3; // metres
    var φ1 = Math.toRadians(lat1);
    var φ2 = Math.toRadians(lat2);
    var Δφ = Math.toRadians((lat2-lat1));
    var Δλ = Math.toRadians((lng2-lng1));

    var a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
            Math.cos(φ1) * Math.cos(φ2) *
            Math.sin(Δλ/2) * Math.sin(Δλ/2);
    var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));

    return R * c;
}

// method adapted from http://www.movable-type.co.uk/scripts/latlong.html
function midpoint(lat1,lng1,lat2,lng2) {
   
    var ptlng = (lng2 - lng1) / 2;
    var ptlat = (lat2 - lat1) / 2;
	return [lng1 + ptlng, lat1 + ptlat];
}


// MongoDB basic query to supplement a Twitter Place with a
// centre-point lat, lng coords for that bounding box,
// and a radius of the long diagonal of that box.
// This could be used to plot a rough point for tweets
// without a geocoded Point already.
db.stream2flood_all.find().limit(5).forEach(
  function(doc){
    
    var lng1 = doc.place.bounding_box.coordinates[0][0][0];
    var lat1 = doc.place.bounding_box.coordinates[0][0][1];
    var lng2 = doc.place.bounding_box.coordinates[0][2][0];
    var lat2 = doc.place.bounding_box.coordinates[0][2][1];
    var obj = {};
    obj["type"] = "Point";
    obj["coordinates"] = midpoint(lat1,lng1,lat2,lng2); 
    
    doc.place_centre_pt = obj;
    doc.place_radius = Math.round(haversine(lat1,lng1,obj["coordinates"][1],obj["coordinates"][0] ) /1000) ; 
    doc.place_pt_lng = obj["coordinates"][0];
    doc.place_pt_lat = obj["coordinates"][1];
    // Debug:
    // print(doc.place.full_name);
    // print(doc.place_radius);
    // print(doc.place.bounding_box.coordinates[0][0][1]);
    // print(doc.place.bounding_box.coordinates[0][0][0]);
    // print(doc.place.bounding_box.coordinates[0][2][1]);
    // print(doc.place.bounding_box.coordinates[0][2][0]);
    // print(doc.place_centre_pt['coordinates'].reverse());
    

    db.stream2flood_all.save(doc);
	}
)


