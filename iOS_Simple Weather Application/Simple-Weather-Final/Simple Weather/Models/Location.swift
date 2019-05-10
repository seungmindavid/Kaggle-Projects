//
//  Location.swift
//  Simple Weather
//
//  Created by Yoshimoto, Kristy Saori on 4/16/19.
//  Copyright © 2019 Yoshimoto, Kristy Saori. All rights reserved.
//

//App Name: Simple Weather
//Turned in: April 21
//Members:
//Kristy Yoshimoto kyoshimo@iu.edu
//Michael McQuitty mmcquitt@iu.edu
//Seung Min Baek seubaek@iu.edu
//Kevin Cao kevcao@iu.edu

import Foundation
import UIKit

class Location: NSObject, Codable {
    
    var name: [String]
    var latitude: [Double]
    var longitude: [Double]
    var currentLocIndex = 0
    
    override var description: String{
        return "\(self.name)\(self.latitude)\(self.longitude)"
    }
    
    init(name: [String], latitude: [Double], longitude: [Double]) {
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
    }
    
    //Returns array of coordinates
    func getLocation() -> [Double]{
        return [self.latitude[currentLocIndex], self.longitude[currentLocIndex]]
    }
    
    //Changes current location index
    func setCurrentLocIndex(newIndex: Int){
        self.currentLocIndex = newIndex
    }
    
    func addNewLocation(name: String, lat: Double, long: Double){
        self.name.append(name)
        self.latitude.append(lat)
        self.longitude.append(long)
    }
    
    func removeLocation(index: Int){
        self.name.remove(at: index)
        self.latitude.remove(at: index)
        self.longitude.remove(at: index)
        if self.currentLocIndex == index {
            if self.currentLocIndex - 1 >= 0 {
                self.currentLocIndex -= 1
            }
        }
    }
    
    //Inserts current location at beginning
    func setCurrentLocation(name: String, lat: Double, long: Double){
        print("in setCurrentLocation")
        self.name.insert("Current Location: \(name)", at: 0)
        self.latitude.insert(lat, at: 0)
        self.longitude.insert(long, at: 0)
    }
    
}
