//
//  LocationSearchTable.swift
//  Simple Weather
//
//  Created by Yoshimoto, Kristy Saori on 4/18/19.
//  Copyright © 2019 Yoshimoto, Kristy Saori. All rights reserved.
//

//App Name: Simple Weather
//Turned in: April 21
//Members:
//Kristy Yoshimoto kyoshimo@iu.edu
//Michael McQuitty mmcquitt@iu.edu
//Seung Min Baek seubaek@iu.edu
//Kevin Cao kevcao@iu.edu

import UIKit
import MapKit

class LocationSearchTable: UITableViewController {
    
    var myAppDelegate: AppDelegate?
    var myModel: Location?

    var match: [MKMapItem] = []
    var mapView: MKMapView? = nil
    
    func parseAddress(selectedItem:MKPlacemark) -> String {
        
        let addressLine = "\(selectedItem.subThoroughfare ?? "") \(selectedItem.thoroughfare ?? ""), \(selectedItem.locality ?? "") \(selectedItem.administrativeArea ?? "")"
        
        return addressLine
    }
    
    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "LocationSearchTableCell") as! LocationSearchTableViewCell
        let selectedItem = match[indexPath.row].placemark
        print(selectedItem)
        cell.nameLabel.text = selectedItem.name
        cell.addressLabel.text = parseAddress(selectedItem: selectedItem)
        return cell
    }
    
    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return match.count
    }
    
    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        self.myAppDelegate = UIApplication.shared.delegate as! AppDelegate
        self.myModel = self.myAppDelegate?.myModel
        
        let place = match[indexPath.row].placemark
        self.myModel?.addNewLocation(name: place.name ?? "", lat: place.coordinate.latitude, long: place.coordinate.longitude)
        
        print(self.myModel?.name)
        
        print("in locationST")
        if let tabVC = UIApplication.shared.keyWindow?.rootViewController as? UITabBarController{
            print("in tabVC")
            if let navVC = tabVC.viewControllers?[2] as? UINavigationController {
                print("in navVC")
                if let setVC = navVC.viewControllers[0] as? SetTabViewController {
                    print("inside setVC")
                    print(setVC)
                    setVC.locations = self.myModel?.name ?? ["no locs...inside LST"]
                    //Clear search bar text once you click on a place
                    setVC.resultSearchController.searchBar.text = nil
                    setVC.tableView.reloadData()
                }
            }
        }
        
        dismiss(animated: true, completion: nil)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
    }
}


extension LocationSearchTable: UISearchResultsUpdating {
    func updateSearchResults(for searchController: UISearchController) {
        
            guard let searchBarText = searchController.searchBar.text else {
                return
            }
            let request = MKLocalSearch.Request()
            request.naturalLanguageQuery = searchBarText
            let search = MKLocalSearch(request: request)
            search.start { response, _ in
                //Array of mapItems
                guard let response = response else {
                    return
                }
                self.match = response.mapItems
                self.tableView.reloadData()
            }
        }
    
}

