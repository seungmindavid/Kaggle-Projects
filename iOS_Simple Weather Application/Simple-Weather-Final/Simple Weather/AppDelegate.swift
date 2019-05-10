//
//  AppDelegate.swift
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

import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?
    
    
    var myModel: Location = Location(name: ["Los Angeles", "Bloomington"], latitude: [34.0522, 39.1636505], longitude: [-118.2437, -86.525757])
    
    var flag1 = false
    var flag2 = false
    let defaults = UserDefaults.standard

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        
        
        print("\n\n ****Inside didFinishLaunchingWithOptions**** \n\n")
        
        self.defaults.register(defaults: ["mapType" : "standard"])
        
        do {
            let fm = FileManager.default
            let docsurl = try fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            
            let myModelFile = docsurl.appendingPathComponent("MyLocation.txt")
            let myModelData = try Data(contentsOf: myModelFile)
            let myModel2 = try PropertyListDecoder().decode(Location.self, from: myModelData)
            
            print("\n\n myModel: \(myModel2) \n\n")
            
            
            let arrFile = docsurl.appendingPathComponent("arrFile.plist")
            let arrayData = try Data(contentsOf: arrFile)
            let arr = try PropertyListDecoder().decode([Location].self, from: arrayData) //An array of flashCardModel
            
            //The model stored in the file should be used
            myModel = arr[0] //Techinically, myLocationModel
            
        } catch {
            flag1 = true
            print(error)
        }
        
        return true
    }

    func applicationWillResignActive(_ application: UIApplication) {
        
        print("\n\n ****Inside applicationWillResignActive**** \n\n")
        
        do {
            let fm = FileManager.default
            let docsurl = try fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            
            
            let myModel2 = Location(name: self.myModel.name, latitude: self.myModel.latitude, longitude: self.myModel.longitude)
            let myModelData = try PropertyListEncoder().encode(myModel2)
            let myModelFile = docsurl.appendingPathComponent("MyLocation.txt")
            
            print("\n\n\n \(myModelFile) \n\n\n")
            print("\n\n\(myModel2)\n\n")
            
            try myModelData.write(to: myModelFile, options: .atomic)
            
            
            
            let arr = [myModel2]
            let arrFile = docsurl.appendingPathComponent("arrFile.plist")
            print(arrFile)
            PropertyListEncoder().outputFormat = .xml
            try PropertyListEncoder().encode(arr).write(to: arrFile, options: .atomic)
            print("we didn't throw writing array of myModel")
            
        } catch {
            flag2 = true
            print(error)
        }
    }

    func applicationDidEnterBackground(_ application: UIApplication) {
        // Use this method to release shared resources, save user data, invalidate timers, and store enough application state information to restore your application to its current state in case it is terminated later.
        // If your application supports background execution, this method is called instead of applicationWillTerminate: when the user quits.
    }

    func applicationWillEnterForeground(_ application: UIApplication) {
        // Called as part of the transition from the background to the active state; here you can undo many of the changes made on entering the background.
    }

    func applicationDidBecomeActive(_ application: UIApplication) {
        // Restart any tasks that were paused (or not yet started) while the application was inactive. If the application was previously in the background, optionally refresh the user interface.
    }

    func applicationWillTerminate(_ application: UIApplication) {
        // Called when the application is about to terminate. Save data if appropriate. See also applicationDidEnterBackground:.
    }


}

