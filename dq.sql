DROP DATABASE IF EXISTS video;
CREATE DATABASE video;
USE video;

CREATE TABLE user (
    id INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(225),
    Email VARCHAR(50),
    Password VARCHAR(50),
    Age INT,
    Mob VARCHAR(50)
);
